#include <sndfile.hh>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <string>
#include <complex>
#include <fftw3.h>
#include <samplerate.h>

// If you want OpenCV post-proc for the non-pseudo path, keep these includes;
// they are not used in pseudo mode.
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ------------------ Biquad + helpers ------------------
struct Biquad {
    double b0=1, b1=0, b2=0, a1=0, a2=0;
    double z1=0, z2=0;
    inline float process(float x) {
        double y = b0*x + z1;
        z1 = b1*x - a1*y + z2;
        z2 = b2*x - a2*y;
        return (float)y;
    }
    inline void reset(){ z1=z2=0; }
    static Biquad bandpass(double fs, double f0, double Q) {
        Biquad biq;
        double w0 = 2.0*M_PI*f0/fs, alpha = sin(w0)/(2.0*Q);
        double b0=  Q*alpha, b1= 0, b2= -Q*alpha;
        double a0=1+alpha, a1=-2*cos(w0), a2=1-alpha;
        biq.b0=b0/a0; biq.b1=b1/a0; biq.b2=b2/a0; biq.a1=a1/a0; biq.a2=a2/a0;
        return biq;
    }
    static Biquad lowpass(double fs, double fc, double Q=0.7071) {
        Biquad biq;
        double w0=2.0*M_PI*fc/fs, alpha=sin(w0)/(2.0*Q);
        double cosw=cos(w0);
        double b0=(1-cosw)/2, b1=1-cosw, b2=(1-cosw)/2;
        double a0=1+alpha, a1=-2*cosw, a2=1-alpha;
        biq.b0=b0/a0; biq.b1=b1/a0; biq.b2=b2/a0; biq.a1=a1/a0; biq.a2=a2/a0;
        return biq;
    }
};

static void filtfilt(std::vector<float>& x, Biquad biq) {
    for (auto& s : x) s = biq.process(s);
    biq.reset();
    for (int i=(int)x.size()-1; i>=0; --i) x[i] = biq.process(x[i]);
}

static double goertzel_power(const float* x, int N, double fs, double f) {
    int k = int(0.5 + (N * f) / fs);
    double w = (2.0 * M_PI / N) * k;
    double cw = 2.0 * cos(w);
    double s0, s1=0.0, s2=0.0;
    for (int n=0; n<N; ++n) {
        s0 = x[n] + cw*s1 - s2;
        s2 = s1; s1 = s0;
    }
    double power = s2*s2 + s1*s1 - cw*s1*s2;
    return power;
}

static std::vector<float> resample_line(const std::vector<float>& seg, int W) {
    std::vector<float> out(W, 0.f);
    if (seg.size() < 2) return out;
    double scale = double(seg.size()-1) / std::max(1, W-1);
    for (int i=0;i<W;++i) {
        double t = i*scale;
        int i0 = std::min<int>((int)seg.size()-2, (int)t);
        double frac = t - i0;
        out[i] = (float)((1.0-frac)*seg[i0] + frac*seg[i0+1]);
    }
    return out;
}

// ------------------ Hilbert FFT envelope ------------------
static std::vector<float> hilbert_envelope_fft(const std::vector<float>& x) {
    int N = (int)x.size();
    if (N<=0) return {};
    std::vector<std::complex<double>> X(N);

    fftw_complex* in  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    for (int i=0;i<N;i++) { in[i][0] = x[i]; in[i][1] = 0.0; }

    fftw_plan fwd = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(fwd);

    for (int k=0; k<N; k++) {
        std::complex<double> val(out[k][0], out[k][1]);
        if (k == 0 || (N%2==0 && k==N/2)) {
            X[k] = val;
        } else if (k < N/2) {
            X[k] = 2.0*val;
        } else {
            X[k] = 0.0;
        }
    }

    for (int i=0;i<N;i++) { in[i][0] = X[i].real(); in[i][1] = X[i].imag(); }
    fftw_plan inv = fftw_plan_dft_1d(N, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(inv);

    std::vector<float> env(N);
    for (int i=0;i<N;i++) {
        double re = out[i][0]/N;
        double im = out[i][1]/N;
        env[i] = (float)std::sqrt(re*re + im*im);
    }

    fftw_destroy_plan(fwd);
    fftw_destroy_plan(inv);
    fftw_free(in);
    fftw_free(out);
    return env;
}

// --------- small utils ----------
static inline float fclamp(float v, float a=0.f, float b=1.f) {
    return std::min(std::max(v, a), b);
}
static inline uint8_t to_u8(float v) {
    return (uint8_t)std::lround(fclamp(v)*255.0f);
}

// ------------------ Main ------------------
int main(int argc, char** argv){
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " input.wav out.png [width=1200] [mode=manual] [color=gray|jet|turbo|hot|bone|ocean|pseudo]\n";
        return 1;
    }
    std::string inpath = argv[1];
    std::string outpng = argv[2];
    int W = (argc>=4)? std::max(200, std::atoi(argv[3])) : 1200;
    std::string mode = (argc>=5)? std::string(argv[4]) : "manual";
    std::string color = (argc>=6)? std::string(argv[5]) : "gray";
    // to lower
    std::transform(mode.begin(), mode.end(), mode.begin(), ::tolower);
    std::transform(color.begin(), color.end(), color.begin(), ::tolower);
    bool wantPseudo = (color == "pseudo");

    // --- Read WAV ---
    SndfileHandle snd(inpath);
    if (snd.error()) { std::cerr << "libsndfile error\n"; return 1; }
    int fs = snd.samplerate();
    int ch = snd.channels();
    sf_count_t N = snd.frames();
    if (N<=0) { std::cerr << "Empty WAV\n"; return 1; }

    std::vector<float> buf(N*ch);
    snd.readf(buf.data(), N);
    std::vector<float> y(N);
    if (ch==1) y = std::move(buf);
    else for (sf_count_t i=0;i<N;++i) y[i] = buf[i*ch + 0];

    // normalize
    float maxv = 0.f;
    for (auto v: y) maxv = std::max(maxv, std::abs(v));
    if (maxv>0) for (auto& v: y) v /= maxv;

    // --- Band-limit around the subcarrier and smooth ---
    Biquad bp = Biquad::bandpass(fs, 2400.0, 1.5);
    std::vector<float> yb = y;
    for (auto& s: yb) s = bp.process(s);
    bp.reset();
    filtfilt(yb, Biquad::lowpass(fs, 4500.0));

    // --- High-quality resampling using libsamplerate ---
    const int DEC = 4;
    int fs_d = fs / DEC;
    double ratio = 1.0 / DEC;
    std::vector<float> yd(yb.size() / DEC + 100);

    SRC_DATA src_data{};
    src_data.data_in = yb.data();
    src_data.input_frames = (long)yb.size();
    src_data.data_out = yd.data();
    src_data.output_frames = (long)yd.size();
    src_data.src_ratio = ratio;
    src_data.end_of_input = 1;

    int err = src_simple(&src_data, SRC_SINC_BEST_QUALITY, 1);
    if (err) { std::cerr << "libsamplerate error: " << src_strerror(err) << "\n"; return 1; }
    yd.resize(src_data.output_frames_gen);

    // Envelope with Hilbert FFT, then light LPF + DC-block
    std::vector<float> env = hilbert_envelope_fft(yd);
    filtfilt(env, Biquad::lowpass(fs_d, 3000.0));
    {
        float px = 0.f, py = 0.f; const float r = 0.995f;
        for (auto& s : env) { float y_ = s - px + r * py; px = s; py = y_; s = y_; }
    }

    // --- Sync detection (as before) ---
    const double f1 = 1040.0, f2 = 832.0;
    const int win_ms = 20, hop_ms = 5;
    const int win = std::max(8, fs_d*win_ms/1000);
    const int hop = std::max(1, fs_d*hop_ms/1000);

    std::vector<double> scores;
    std::vector<int> centers;
    std::vector<float> seg(win);
    for (int start=0; start+win<(int)env.size(); start+=hop) {
        for (int i=0;i<win;++i){
            double w = 0.5 - 0.5*cos(2.0*M_PI*i/(win-1));
            seg[i] = (float)(env[start+i]*w);
        }
        double p = goertzel_power(seg.data(), win, fs_d, f1) +
                   goertzel_power(seg.data(), win, fs_d, f2);
        scores.push_back(p);
        centers.push_back(start + win/2);
    }
    double minsc=*std::min_element(scores.begin(), scores.end());
    double maxsc=*std::max_element(scores.begin(), scores.end());
    if (maxsc>minsc) for (auto& s: scores) s = (s - minsc) / (maxsc - minsc);

    int minDistHops = int(0.40 * fs_d / hop);
    std::vector<int> peaks;
    double thr = 0.2;
    int last = -minDistHops;
    for (int i=0;i<(int)scores.size();++i){
        if (scores[i] >= thr && i - last >= minDistHops) {
            int lo = std::max(0, i-3), hi = std::min((int)scores.size()-1, i+3);
            int argmax = i;
            for (int k=lo;k<=hi;++k) if (scores[k] > scores[argmax]) argmax=k;
            if (argmax==i) { peaks.push_back(centers[i]); last = i; }
        }
    }
    std::vector<int> line_starts;
    for (size_t i=0;i<peaks.size();++i){
        if (i==0 || (peaks[i]-peaks[i-1]) > int(0.20*fs_d)) line_starts.push_back(peaks[i]);
    }
    if (line_starts.size() < 10) { std::cerr << "Too few line starts.\n"; return 1; }

    // --- Build per-line data ---
    // If pseudo color requested: collect VIS/IR float rows
    // Else: collect grayscale rows (uint8) as before
    std::vector<std::vector<float>> visRowsF, irRowsF;
    std::vector<std::vector<uint8_t>> grayRows;

    // Heuristic layout fractions: skip sync/porch, then split remaining into VIS/IR halves
    const double skip_frac = 0.08; // ~8% of line to skip (sync + black porch)
    for (size_t i=0;i+1<line_starts.size();++i){
        int s0 = line_starts[i];
        int s1 = line_starts[i+1];
        if (s1 <= s0 + int(0.2*fs_d)) continue;

        std::vector<float> segline(env.begin()+s0, env.begin()+s1);

        // robust percentile clip per line (1–99%)
        std::vector<float> tmp = segline;
        std::nth_element(tmp.begin(), tmp.begin()+ (int)tmp.size()/100, tmp.end());
        float lo = tmp[(int)tmp.size()/100];
        std::nth_element(tmp.begin(), tmp.begin()+ (int)tmp.size()*99/100, tmp.end());
        float hi = tmp[(int)tmp.size()*99/100];
        if (hi <= lo) continue;
        for (auto& v: segline){ v = std::clamp((v - lo) / (hi - lo), 0.0f, 1.0f); }

        // Running-median style per-line gain normalization (as before)
        double mean = 0.0;
        for (auto v : segline) mean += v;
        mean /= std::max<size_t>(1, segline.size());
        static std::vector<double> recentMeans;
        recentMeans.push_back(mean);
        size_t Wm = 101;
        if (recentMeans.size() > Wm) recentMeans.erase(recentMeans.begin());
        std::vector<double> sorted = recentMeans;
        std::nth_element(sorted.begin(), sorted.begin() + (int)sorted.size()/2, sorted.end());
        double targetMed = sorted[(int)sorted.size()/2];
        double gainMed = (mean > 1e-6) ? (targetMed / mean) : 1.0;
        gainMed = std::clamp(gainMed, 0.7, 1.4);
        for (auto& v : segline) v = std::clamp(float(v * gainMed), 0.0f, 1.0f);

        if (wantPseudo) {
            // --- split into VIS (first half after skip) and IR (second half) ---
            int n = (int)segline.size();
            int start = std::clamp((int)std::round(n*skip_frac), 0, n);
            int rem = n - start;
            if (rem < 20) continue;
            int half = rem/2;
            std::vector<float> segVIS(segline.begin()+start, segline.begin()+start+half);
            std::vector<float> segIR (segline.begin()+start+half, segline.begin()+start+2*half);

            auto visL = resample_line(segVIS, W);
            auto irL  = resample_line(segIR,  W);
            visRowsF.push_back(std::move(visL));
            irRowsF .push_back(std::move(irL));
        } else {
            // grayscale path (unchanged)
            auto line = resample_line(segline, W);
            std::vector<uint8_t> row(W);
            for (int j=0;j<W;++j) row[j] = (uint8_t)std::lround(std::clamp(line[j]*255.0f, 0.0f, 255.0f));
            grayRows.push_back(std::move(row));
        }
    }

    int H = wantPseudo ? (int)visRowsF.size() : (int)grayRows.size();
    if (H < 10) { std::cerr << "Too few rows.\n"; return 1; }

    if (!wantPseudo) {
        // ----- legacy grayscale (optionally you can add OpenCV here like before) -----
        std::vector<uint8_t> img(H*W);
        for (int r=0;r<H;++r) std::copy(grayRows[r].begin(), grayRows[r].end(), img.begin()+r*W);

        // If you want OpenCV CLAHE+denoise (non-pseudo), you can drop it here using img -> cv::Mat
        if (!stbi_write_png(outpng.c_str(), W, H, 1, img.data(), W)) {
            std::cerr << "Failed to write PNG\n"; return 1;
        }
        std::cerr << "Saved " << outpng << " (" << W << "x" << H << ") grayscale\n";
        return 0;
    }

    // ----- PSEUDO COLOR COMPOSITE -----
    // 1) Gather global percentiles for VIS and IR to normalize channels robustly
    std::vector<float> allVIS; allVIS.reserve((size_t)H * W);
    std::vector<float> allIR;  allIR .reserve((size_t)H * W);
    for (int r=0;r<H;++r) {
        allVIS.insert(allVIS.end(), visRowsF[r].begin(), visRowsF[r].end());
        allIR .insert(allIR .end(), irRowsF [r].begin(), irRowsF [r].end());
    }
    auto pct = [](std::vector<float>& v, double p){
        if (v.empty()) return 0.f;
        size_t k = (size_t)std::clamp((int)std::round((p/100.0)*(v.size()-1)), 0, (int)v.size()-1);
        std::nth_element(v.begin(), v.begin()+k, v.end());
        return v[k];
    };
    std::vector<float> tmpVIS = allVIS, tmpIR = allIR;
    float vis_lo = pct(tmpVIS, 1.0), vis_hi = pct(tmpVIS, 99.0);
    float ir_lo  = pct(tmpIR,  1.0), ir_hi  = pct(tmpIR,  99.0);
    float eps = 1e-6f;

    auto norm = [&](float v, float lo, float hi){
        return fclamp((v - lo) / std::max(hi - lo, eps));
    };

    // 2) Compose RGB using VIS luminance and NDVI-like cue (IR vs VIS)
    std::vector<uint8_t> rgb((size_t)H*W*3);
    for (int r=0;r<H;++r) {
        for (int c=0;c<W;++c) {
            float vis = norm(visRowsF[r][c], vis_lo, vis_hi);
            float ir  = norm(irRowsF [r][c], ir_lo , ir_hi );

            // NDVI-like: [-1..1], then to [0..1]
            float ndvi = (ir - vis) / (ir + vis + eps);
            float vmask = fclamp(0.5f * (ndvi + 1.0f)); // vegetation mask in [0..1]

            // Luminance base from VIS
            float L = vis;

            // Heuristic mapping:
            // - vegetation (high vmask) → more green
            // - ocean (low L, low vmask) → deeper blue
            // - clouds (high L, low |ndvi|) stay white-ish
            float R = fclamp(0.50f*L + 0.80f*vmask);
            float G = fclamp(0.90f*L + 0.20f*vmask);
            float B = fclamp(1.00f*L - 0.50f*vmask + 0.10f);

            size_t idx = ((size_t)r*W + c)*3;
            rgb[idx+0] = to_u8(R);
            rgb[idx+1] = to_u8(G);
            rgb[idx+2] = to_u8(B);
        }
    }

    // 3) Write RGB PNG
    int stride = W*3;
    if (!stbi_write_png(outpng.c_str(), W, H, 3, rgb.data(), stride)) {
        std::cerr << "Failed to write PNG (pseudo)\n"; return 1;
    }
    std::cerr << "Saved " << outpng << " (" << W << "x" << H << ") pseudo-color (VIS+IR)\n";
    return 0;
}
