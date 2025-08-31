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

// Zero-phase filter by forward/backward pass
static void filtfilt(std::vector<float>& x, Biquad biq) {
    for (auto& s : x) s = biq.process(s);
    biq.reset();
    for (int i=(int)x.size()-1; i>=0; --i) x[i] = biq.process(x[i]);
}

// Goertzel (power at a single frequency)
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

// Linear resample to fixed length
static std::vector<float> resample_line(const std::vector<float>& seg, int W) {
    std::vector<float> out(W, 0.f);
    if (seg.size() < 2) return out;
    double scale = double(seg.size()-1) / (W-1);
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
    std::vector<std::complex<double>> X(N);

    fftw_complex* in  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);

    for (int i=0;i<N;i++) { in[i][0] = x[i]; in[i][1] = 0.0; }

    fftw_plan fwd = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(fwd);

    for (int k=0; k<N; k++) {
        std::complex<double> val(out[k][0], out[k][1]);
        if (k == 0 || (N%2==0 && k==N/2)) {
            X[k] = val; // keep DC and Nyquist
        } else if (k < N/2) {
            X[k] = 2.0*val; // positive freqs
        } else {
            X[k] = 0.0;     // negative freqs
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

// --- Multi-level Haar wavelet denoising (for "manual" mode) ---
static void haar_dwt(std::vector<double>& data, int levels) {
    int n = (int)data.size();
    for (int lev=0; lev<levels; lev++) {
        int step = n >> lev;
        if (step < 2) break;
        std::vector<double> approx(step/2), detail(step/2);
        for (int i=0; i<step/2; i++) {
            approx[i] = (data[2*i] + data[2*i+1]) / 2.0;
            detail[i] = (data[2*i] - data[2*i+1]) / 2.0;
        }
        for (int i=0; i<step/2; i++) {
            data[i] = approx[i];
            data[step/2 + i] = detail[i];
        }
    }
}

static void haar_idwt(std::vector<double>& data, int levels) {
    int n = (int)data.size();
    for (int lev=levels-1; lev>=0; lev--) {
        int step = n >> lev;
        if (step < 2) continue;
        std::vector<double> approx(step/2), detail(step/2);
        for (int i=0; i<step/2; i++) {
            approx[i] = data[i];
            detail[i] = data[step/2 + i];
        }
        for (int i=0; i<step/2; i++) {
            data[2*i]   = approx[i] + detail[i];
            data[2*i+1] = approx[i] - detail[i];
        }
    }
}

static std::vector<uint8_t> haar_denoise_multi(const std::vector<uint8_t>& row, int levels, double thresh) {
    int n = (int)row.size();
    std::vector<double> data(row.begin(), row.end());

    haar_dwt(data, levels);

    int offset = 0;
    for (int lev=0; lev<levels; lev++) {
        int step = n >> lev;
        int half = step/2;
        offset += half; // details start here
        for (int i=0; i<half; i++) {
            double& d = data[offset + i];
            if (std::abs(d) < thresh) d = 0;
            else d = (d > 0 ? d-thresh : d+thresh); // soft threshold
        }
    }

    haar_idwt(data, levels);

    std::vector<uint8_t> out(n);
    for (int i=0; i<n; i++) {
        out[i] = (uint8_t)std::clamp(std::round(data[i]), 0.0, 255.0);
    }
    return out;
}

// ---- Colormap name -> OpenCV enum (returns -1 for gray) ----
static int map_colormap(const std::string& name) {
    std::string s = name;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return (char)std::tolower(c); });
    if (s=="gray" || s=="greyscale" || s=="grayscale" || s=="none") return -1;
    if (s=="jet")      return cv::COLORMAP_JET;
    if (s=="ocean")    return cv::COLORMAP_OCEAN;
    //if (s=="terrain")  return cv::COLORMAP_TERRAIN;
    if (s=="hot")      return cv::COLORMAP_HOT;
    if (s=="bone")     return cv::COLORMAP_BONE;
    if (s=="winter")   return cv::COLORMAP_WINTER;
    if (s=="rainbow")  return cv::COLORMAP_RAINBOW;
    if (s=="autumn")   return cv::COLORMAP_AUTUMN;
    if (s=="summer")   return cv::COLORMAP_SUMMER;
    if (s=="spring")   return cv::COLORMAP_SPRING;
    if (s=="cool")     return cv::COLORMAP_COOL;
    if (s=="hsv")      return cv::COLORMAP_HSV;
    if (s=="pink")     return cv::COLORMAP_PINK;
    if (s=="parula")   return cv::COLORMAP_PARULA;
#ifdef CV_COLORMAP_TURBO
    if (s=="turbo")    return cv::COLORMAP_TURBO;
#else
    if (s=="turbo")    return cv::COLORMAP_JET; // fallback
#endif
    return -1; // default to gray if unknown
}

// ------------------ Main ------------------
int main(int argc, char** argv){
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " input.wav out.png [width=1200] [mode] [color]\n";
        std::cerr << "Modes: opencv (default) | manual\n";
        std::cerr << "Color: gray (default) | jet | ocean | hot | bone | winter | rainbow | autumn | summer | spring | cool | hsv | pink | parula | turbo\n";
        return 1;
    }
    std::string inpath = argv[1];
    std::string outpng = argv[2];
    int W = (argc>=4)? std::max(200, std::atoi(argv[3])) : 1200;
    std::string mode  = (argc>=5)? std::string(argv[4]) : std::string("opencv"); // default
    std::string color = (argc>=6)? std::string(argv[5]) : std::string("gray");   // default

    // --- Read WAV (libsndfile) ---
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

    // --- Band-limit around subcarrier (~2.4 kHz) ---
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

    SRC_DATA src_data;
    src_data.data_in = yb.data();
    src_data.input_frames = (long)yb.size();
    src_data.data_out = yd.data();
    src_data.output_frames = (long)yd.size();
    src_data.src_ratio = ratio;
    src_data.end_of_input = 1;

    int err = src_simple(&src_data, SRC_SINC_BEST_QUALITY, 1);
    if (err) {
        std::cerr << "libsamplerate error: " << src_strerror(err) << "\n";
        return 1;
    }
    yd.resize(src_data.output_frames_gen);

    // --- Envelope with Hilbert FFT ---
    std::vector<float> env = hilbert_envelope_fft(yd);
    filtfilt(env, Biquad::lowpass(fs_d, 3000.0));

    // --- DC-block ---
    {
        float px = 0.f, py = 0.f;
        const float r = 0.995f;
        for (auto& s : env) {
            float y_ = s - px + r * py;
            px = s; py = y_;
            s = y_;
        }
    }

    // --- Sync detection (Goertzel on sliding windows) ---
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

    // --- Assemble image rows with per-line normalization ---
    std::vector<std::vector<uint8_t>> rows;
    rows.reserve(line_starts.size());
    static std::vector<double> recentMeans; // for running-median AGC
    for (size_t i=0;i+1<line_starts.size();++i){
        int s0 = line_starts[i];
        int s1 = line_starts[i+1];
        if (s1 <= s0 + int(0.2*fs_d)) continue;
        std::vector<float> segline(env.begin()+s0, env.begin()+s1);

        // percentile clip (1–99%)
        std::vector<float> tmp = segline;
        std::nth_element(tmp.begin(), tmp.begin()+ (int)tmp.size()/100, tmp.end());
        float lo = tmp[(int)tmp.size()/100];
        std::nth_element(tmp.begin(), tmp.begin()+ (int)tmp.size()*99/100, tmp.end());
        float hi = tmp[(int)tmp.size()*99/100];
        if (hi <= lo) continue;
        for (auto& v: segline){ v = std::clamp((v - lo) / (hi - lo), 0.0f, 1.0f); }

        // running-median per-row normalization
        double mean = 0.0;
        for (auto v : segline) mean += v;
        mean /= std::max<size_t>(1, segline.size());
        recentMeans.push_back(mean);
        size_t Wm = 101;
        if (recentMeans.size() > Wm) recentMeans.erase(recentMeans.begin());
        std::vector<double> sorted = recentMeans;
        std::nth_element(sorted.begin(), sorted.begin() + (int)sorted.size()/2, sorted.end());
        double targetMed = sorted[(int)sorted.size()/2];
        double gainMed = (mean > 1e-6) ? (targetMed / mean) : 1.0;
        gainMed = std::clamp(gainMed, 0.7, 1.4);
        for (auto& v : segline) v = std::clamp(float(v * gainMed), 0.0f, 1.0f);

        auto line = resample_line(segline, W);
        std::vector<uint8_t> row(W);
        for (int j=0;j<W;++j) row[j] = (uint8_t)std::lround(std::clamp(line[j]*255.0f, 0.0f, 255.0f));

        rows.push_back(std::move(row));
    }
    if (rows.empty()) { std::cerr << "No rows assembled.\n"; return 1; }

    const int H = (int)rows.size();

    // --------- BRANCH: manual vs OpenCV ----------
    std::vector<uint8_t> grayBuf; // final grayscale before optional color
    if (mode == "manual") {
        // Haar denoise each row
        for (auto& row : rows) {
            row = haar_denoise_multi(row, 3, 15.0); // levels, threshold
        }
        grayBuf.resize(H*W);
        for (int r=0;r<H;++r)
            std::copy(rows[r].begin(), rows[r].end(), grayBuf.begin()+r*W);

        // Global histogram equalization
        const int levels = 256;
        std::vector<int> hist(levels, 0);
        for (auto v : grayBuf) hist[v]++;
        std::vector<int> cdf(levels, 0);
        cdf[0] = hist[0];
        for (int i=1;i<levels;i++) cdf[i] = cdf[i-1] + hist[i];
        int total = H * W;
        int cdf_min = 0;
        for (int i=0;i<levels;i++) { if (cdf[i] != 0) { cdf_min = cdf[i]; break; } }
        std::vector<uint8_t> lut(levels);
        for (int i=0;i<levels;i++) {
            lut[i] = (uint8_t)std::round(((double)(cdf[i] - cdf_min) / (std::max(1, total - cdf_min))) * 255.0);
        }
        for (auto& v : grayBuf) v = lut[v];

    } else {
        // OpenCV CLAHE + NLM
        grayBuf.resize(H*W);
        for (int r=0;r<H;++r)
            std::copy(rows[r].begin(), rows[r].end(), grayBuf.begin()+r*W);

        cv::Mat imgMat(H, W, CV_8UC1, grayBuf.data());
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(2.0);
        clahe->setTilesGridSize(cv::Size(8, 8));
        cv::Mat imgClahe;
        clahe->apply(imgMat, imgClahe);

        cv::Mat imgDenoised;
        cv::fastNlMeansDenoising(imgClahe, imgDenoised, 10, 7, 21);

        // overwrite grayBuf with denoised result
        grayBuf.assign(imgDenoised.begin<uint8_t>(), imgDenoised.end<uint8_t>());
    }

    // --------- Optional color mapping with OpenCV ---------
    int cmap = map_colormap(color);
    if (cmap < 0) {
        // Gray output
        if (!stbi_write_png(outpng.c_str(), W, H, 1, grayBuf.data(), W)) {
            std::cerr << "Failed to write PNG\n"; return 1;
        }
        std::cerr << "Saved " << outpng << " (" << W << "x" << H << ") mode=" << mode << " color=gray\n";
    } else {
        cv::Mat grayMat(H, W, CV_8UC1, grayBuf.data());
        if (!grayMat.isContinuous()) grayMat = grayMat.clone(); // ensure contiguous

        cv::Mat bgr;
        cv::applyColorMap(grayMat, bgr, cmap);         // BGR, CV_8UC3
        if (!bgr.data) { std::cerr << "applyColorMap produced empty image\n"; return 1; }

        cv::Mat rgb;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);     // RGB for stbi
        if (!rgb.isContinuous()) rgb = rgb.clone();    // ensure contiguous

        // Use Mat's stride (bytes per row). stb expects row stride in bytes.
        int stride = static_cast<int>(rgb.step);
        if (!stbi_write_png(outpng.c_str(), W, H, 3, rgb.data, stride)) {
            std::cerr << "Failed to write PNG (color)\n"; return 1;
        }
        std::cerr << "Saved " << outpng << " (" << W << "x" << H << ") mode=" << mode << " color=" << color << "\n";
    }


    return 0;
}