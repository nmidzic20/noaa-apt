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
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
    for (auto& s : x) s = biq.process(s);           // forward
    biq.reset();
    for (int i=(int)x.size()-1; i>=0; --i) x[i] = biq.process(x[i]); // reverse
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
    std::vector<float> out(W);
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

static std::vector<float> decimate(const std::vector<float>& x, int M) {
    std::vector<float> y;
    y.reserve(x.size()/M + 1);
    for (size_t i=0; i<x.size(); i += M) y.push_back(x[i]);
    return y;
}


static std::vector<float> hilbert_envelope_fir(const std::vector<float>& x) {
    constexpr int N = 65;                  
    constexpr int M = (N - 1) / 2;         
    static float h[N];
    static bool inited = false;

    if (!inited) {
        for (int n=0; n<N; ++n) {
            int k = n - M;
            float val = 0.0f;
            if (k != 0 && (k & 1)) {       
                val = 2.0f / (float)(M_PI * k);
            }
            // Hamming window to reduce ripple
            float w = 0.54f - 0.46f * std::cos(2.0f * (float)M_PI * n / (N-1));
            h[n] = val * w;
        }
        inited = true;
    }

    const int L = (int)x.size();
    std::vector<float> imag(L, 0.f), env;
    env.reserve(L);

    // Convolve x with h -> imag (causal, zero-padded)
    for (int n=0; n<L; ++n) {
        double acc = 0.0;
        int kmin = std::max(0, n - (L-1)); 
        int kmax = std::min(N-1, n);
        (void)kmin;
        for (int k=0; k<=kmax; ++k) {
            int m = n - k;
            acc += x[m] * h[k];
        }
        imag[n] = (float)acc;
    }

    // Real path delayed by M
    for (int n=0; n<L; ++n) {
        float re = (n >= M) ? x[n - M] : 0.0f;
        float im = imag[n];
        env.push_back(std::sqrt(re*re + im*im));
    }
    return env;
}

int main(int argc, char** argv){
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " input.wav out.png [width=1200]\n";
        return 1;
    }
    std::string inpath = argv[1];
    std::string outpng = argv[2];
    int W = (argc>=4)? std::max(200, std::atoi(argv[3])) : 1200;

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
        for (sf_count_t i=0;i<N;++i) y[i] = buf[i*ch + 0]; 
    }
    float maxv = 0.f;
    for (auto v: y) maxv = std::max(maxv, std::abs(v));
    if (maxv>0) for (auto& v: y) v /= maxv;


    Biquad bp = Biquad::bandpass(fs, 2400.0, 1.5);
    std::vector<float> yb = y;
    for (auto& s: yb) s = bp.process(s);
    bp.reset();
    filtfilt(yb, Biquad::lowpass(fs, 4500.0));  

    const int DEC = 4;
    filtfilt(yb, Biquad::lowpass(fs, fs/(2.0*DEC) * 0.9)); 
    std::vector<float> yd = decimate(yb, DEC);
    int fs_d = fs / DEC;

    std::vector<float> env = hilbert_envelope_fir(yd);

    filtfilt(env, Biquad::lowpass(fs_d, 3000.0));

    {
        float px = 0.f, py = 0.f;
        const float r = 0.995f;
        for (auto& s : env) {
            float y_ = s - px + r * py;
            px = s; py = y_;
            s = y_;
        }
    }

    const double f1 = 1040.0, f2 = 832.0; // NOAA APT sync tones
    const int win_ms = 20, hop_ms = 5;
    const int win = std::max(8, fs_d*win_ms/1000);
    const int hop = std::max(1, fs_d*hop_ms/1000);

    std::vector<double> scores;
    std::vector<int> centers;
    std::vector<float> seg(win);
    for (int start=0; start+win<(int)env.size(); start+=hop) {
        // Hanning window
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
    if (maxsc>minsc) {
        for (auto& s: scores) s = (s - minsc) / (maxsc - minsc);
    } else {
        std::cerr << "Flat scores; sync not found.\n"; return 1;
    }

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
    if (line_starts.size() < 10) {
        std::cerr << "Too few line starts: " << line_starts.size() << "\n";
        return 1;
    }
    std::cerr << "Detected lines: " << line_starts.size() << "\n";

    std::vector<std::vector<uint8_t>> rows;
    rows.reserve(line_starts.size());
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

        auto line = resample_line(segline, W);
        std::vector<uint8_t> row(W);
        for (int j=0;j<W;++j) row[j] = (uint8_t)std::lround(std::clamp(line[j]*255.0f, 0.0f, 255.0f));
        rows.push_back(std::move(row));
    }
    if (rows.size() < 10) { std::cerr << "Too few rows.\n"; return 1; }

    int H = (int)rows.size();
    std::vector<uint8_t> img(H*W);
    for (int r=0;r<H;++r) std::copy(rows[r].begin(), rows[r].end(), img.begin()+r*W);

    auto med3 = [&](int r, int c){
        uint8_t k[9]; int idx=0;
        for (int dr=-1; dr<=1; ++dr)
          for (int dc=-1; dc<=1; ++dc){
            int rr = std::clamp(r+dr, 0, H-1);
            int cc = std::clamp(c+dc, 0, W-1);
            k[idx++] = img[rr*W+cc];
          }
        std::nth_element(k, k+4, k+9);
        return k[4];
    };
    {
        std::vector<uint8_t> out(img.size());
        for (int r=0;r<H;++r)
          for (int c=0;c<W;++c)
            out[r*W+c] = med3(r,c);
        img.swap(out);
    }

    if (!stbi_write_png(outpng.c_str(), W, H, 1, img.data(), W)) {
        std::cerr << "Failed to write PNG\n"; return 1;
    }
    std::cerr << "Saved " << outpng << " (" << H << "x" << W << ")\n";
    return 0;
}
