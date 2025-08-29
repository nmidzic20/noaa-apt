#include <sndfile.hh>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <string>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Biquad {
    // RBJ audio EQ cookbook biquad
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
        // constant skirt gain, peak gain = Q
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

// Zero-phase filter by forward/backward pass (like filtfilt)
static void filtfilt(std::vector<float>& x, Biquad biq) {
    // forward
    for (auto& s : x) s = biq.process(s);
    biq.reset();
    // reverse
    for (int i=(int)x.size()-1; i>=0; --i) x[i] = biq.process(x[i]);
}

// Simple Goertzel (power at a single frequency)
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
    std::vector<float> out(W);
    double scale = double(seg.size()-1) / (W-1);
    for (int i=0;i<W;++i) {
        double t = i*scale;
        int i0 = std::min<int>(seg.size()-2, (int)t);
        double frac = t - i0;
        out[i] = (float)((1.0-frac)*seg[i0] + frac*seg[i0+1]);
    }
    return out;
}

int main(int argc, char** argv){
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " input.wav out.png [width=1200]\n";
        return 1;
    }
    std::string inpath = argv[1];
    std::string outpng = argv[2];
    int W = (argc>=4)? std::max(200, std::atoi(argv[3])) : 1200;

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
    if (ch==1) {
        y = std::move(buf);
    } else {
        // take channel 0
        for (sf_count_t i=0;i<N;++i) y[i] = buf[i*ch + 0];
    }
    // normalize
    float maxv = 0.f;
    for (auto v: y) maxv = std::max(maxv, std::abs(v));
    if (maxv>0) for (auto& v: y) v /= maxv;

    // --- Band-pass ~1.5–3.5 kHz using two cascaded biquads centered at 2.4 kHz ---
    Biquad bp = Biquad::bandpass(fs, 2400.0, 1.5); // gentler
    std::vector<float> yb = y;
    for (auto& s: yb) s = bp.process(s);
    bp.reset();
    filtfilt(yb, Biquad::lowpass(fs, 4500.0));     // touch of smoothing pre-envelope

    // --- Envelope via rectifier + low-pass ---
    for (auto& s: yb) s = std::abs(s);
    filtfilt(yb, Biquad::lowpass(fs, 2500.0)); // smooth video

    // --- Sync detection (Goertzel on sliding windows) ---
    const double f1 = 1040.0, f2 = 832.0; // NOAA APT sync tones
    const int win_ms = 20, hop_ms = 5;
    const int win = std::max(8, fs*win_ms/1000);
    const int hop = std::max(1, fs*hop_ms/1000);

    std::vector<double> scores;
    std::vector<int> centers;
    std::vector<float> seg(win);
    for (int start=0; start+win<(int)yb.size(); start+=hop) {
        // Hanning window
        for (int i=0;i<win;++i){
            double w = 0.5 - 0.5*cos(2.0*M_PI*i/(win-1));
            seg[i] = (float)(yb[start+i]*w);
        }
        double p = goertzel_power(seg.data(), win, fs, f1) +
                   goertzel_power(seg.data(), win, fs, f2);
        scores.push_back(p);
        centers.push_back(start + win/2);
    }
    // Normalize scores
    double minsc=*std::min_element(scores.begin(), scores.end());
    double maxsc=*std::max_element(scores.begin(), scores.end());
    if (maxsc>minsc) {
        for (auto& s: scores) s = (s - minsc) / (maxsc - minsc);
    } else {
        std::cerr << "Flat scores; sync not found.\n"; return 1;
    }

    // Peak picking with refractory period ~0.4 s
    int minDistHops = int(0.40 * fs / hop);
    std::vector<int> peaks;
    double thr = 0.2;
    int last = -minDistHops;
    for (int i=0;i<(int)scores.size();++i){
        if (scores[i] >= thr && i - last >= minDistHops) {
            // local max check in small neighborhood
            int lo = std::max(0, i-3), hi = std::min((int)scores.size()-1, i+3);
            int argmax = i;
            for (int k=lo;k<=hi;++k) if (scores[k] > scores[argmax]) argmax=k;
            if (argmax==i) { peaks.push_back(centers[i]); last = i; }
        }
    }
    // Drop too-close peaks (<0.2 s apart)
    std::vector<int> line_starts;
    for (size_t i=0;i<peaks.size();++i){
        if (i==0 || (peaks[i]-peaks[i-1]) > int(0.20*fs)) line_starts.push_back(peaks[i]);
    }
    if (line_starts.size() < 10) {
        std::cerr << "Too few line starts: " << line_starts.size() << "\n";
        return 1;
    }
    std::cerr << "Detected lines: " << line_starts.size() << "\n";

    // --- Assemble image with per-line deskew + robust per-line AGC ---
    std::vector<std::vector<uint8_t>> rows;
    rows.reserve(line_starts.size());
    for (size_t i=0;i+1<line_starts.size();++i){
        int s0 = line_starts[i];
        int s1 = line_starts[i+1];
        if (s1 <= s0 + int(0.2*fs)) continue;
        std::vector<float> segline(yb.begin()+s0, yb.begin()+s1);

        // percentile clip (1–99%)
        std::vector<float> tmp = segline;
        std::nth_element(tmp.begin(), tmp.begin()+tmp.size()/100, tmp.end());
        float lo = tmp[tmp.size()/100];
        std::nth_element(tmp.begin(), tmp.begin()+tmp.size()*99/100, tmp.end());
        float hi = tmp[tmp.size()*99/100];
        if (hi <= lo) continue;
        for (auto& v: segline){ v = std::clamp((v - lo) / (hi - lo), 0.0f, 1.0f); }

        // Row-mean normalization to remove horizontal bands
        double mean = 0.0;
        for (auto v : segline) mean += v;
        mean /= std::max<size_t>(1, segline.size());

        static std::vector<double> recentMeans;
        recentMeans.push_back(mean);
        size_t Wm = 51; // window of ~51 lines
        if (recentMeans.size() > Wm) recentMeans.erase(recentMeans.begin());
        std::vector<double> sorted = recentMeans;
        std::nth_element(sorted.begin(), sorted.begin() + sorted.size()/2, sorted.end());
        double targetMed = sorted[sorted.size()/2];
        double gainMed = (mean > 1e-6) ? (targetMed / mean) : 1.0;
        for (auto& v : segline) v = std::clamp(float(v * gainMed), 0.0f, 1.0f);


        auto line = resample_line(segline, W);
        std::vector<uint8_t> row(W);
        for (int j=0;j<W;++j) row[j] = (uint8_t)std::lround(std::clamp(line[j]*255.0f, 0.0f, 255.0f));
        rows.push_back(std::move(row));
    }
    if (rows.size() < 10) { std::cerr << "Too few rows.\n"; return 1; }

    int H = (int)rows.size();
    std::vector<uint8_t> img(H*W);
    for (int r=0;r<H;++r) std::copy(rows[r].begin(), rows[r].end(), img.begin()+r*W);

    if (!stbi_write_png(outpng.c_str(), W, H, 1, img.data(), W)) {
        std::cerr << "Failed to write PNG\n"; return 1;
    }
    std::cerr << "Saved " << outpng << " (" << H << "x" << W << ")\n";
    return 0;
}
