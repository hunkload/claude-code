import numpy as np, wave, subprocess, sys, math, os
from multiprocessing import Pool
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUT = sys.argv[1]
SR = 44100; BPM = 96; BEAT = 60/BPM; BAR = BEAT*4; NBARS = 20
DUR = BAR*NBARS; N = int(DUR*SR)
rng = np.random.default_rng(7)

# ---------------- AUDIO ----------------
def lp(x, taps=101, fc=0.05):
    n = np.arange(taps)-(taps-1)/2
    k = np.sinc(2*fc*n)*np.hamming(taps); k /= k.sum()
    return np.convolve(x, k, mode='same')
def env(n, d): return np.exp(-np.arange(n)/(d*SR))
mix = np.zeros(N); bassbus = np.zeros(N); sc = np.ones(N)
def add(buf, start, sig):
    i = int(start*SR); j = min(N, i+len(sig))
    if i < N: buf[i:j] += sig[:j-i]

def kick():
    n = int(0.45*SR); t = np.arange(n)/SR
    f = 45 + 110*np.exp(-t*28); ph = 2*np.pi*np.cumsum(f)/SR
    return np.tanh(3*np.sin(ph)*env(n, .18))*0.9
def snare():
    n = int(0.3*SR); t = np.arange(n)/SR
    nz = np.diff(rng.standard_normal(n+1))*0.5
    s = (nz*env(n, .07) + 0.5*np.sin(2*np.pi*185*t)*env(n, .05))
    return np.round(np.tanh(2.5*s)*12)/12*0.6   # bitcrushed grit
def hat(o=False):
    n = int((0.25 if o else 0.05)*SR)
    return np.diff(np.diff(rng.standard_normal(n+2)))*env(n, .08 if o else .012)*0.07
def beep(f, d=0.12):
    n = int(d*SR); t = np.arange(n)/SR
    s = np.sign(np.sin(2*np.pi*f*t))*0.3 + np.sin(2*np.pi*f*t)*0.7
    a = np.minimum(1, t/0.003)*np.exp(-t/(d*0.6))
    return lp(s*a, 31, 0.15)*0.22
def heartbeat():
    n = int(0.35*SR); t = np.arange(n)/SR
    lub = np.sin(2*np.pi*55*t)*np.exp(-t/0.05)
    dub = np.zeros(n); k = int(0.16*SR); tt = t[:n-k]
    dub[k:] = 0.7*np.sin(2*np.pi*65*tt)*np.exp(-tt/0.04)
    return np.tanh(2*(lub+dub))*0.8

K, S = kick(), snare()
roots = [55.0, 43.65, 65.41, 49.0]   # A1 F1 C2 G1
lead = [880, 1046.5, 1318.5, 1174.7, 987.8, 880, 784, 659.3]
events = {'kick': [], 'snare': [], 'beep': []}

def section(bar):
    if bar < 4: return 'intro'
    if bar < 10: return 'verse'
    if bar < 17: return 'drop'
    if bar == 17: return 'flat'
    return 'outro'

for bar in range(NBARS):
    sec = section(bar); b0 = bar*BAR
    for beat in range(4):
        t = b0 + beat*BEAT
        if sec == 'intro':
            add(mix, t, heartbeat()); events['kick'].append(t)
        elif sec in ('verse', 'drop', 'outro'):
            add(mix, t, K); events['kick'].append(t)
            e = 1-0.85*np.exp(-np.arange(int(BEAT*SR))/(0.09*SR))
            i = int(t*SR); sc[i:i+len(e)] = e[:N-i]
            if beat in (1, 3):
                add(mix, t, S); events['snare'].append(t)
            for h in range(2 if sec == 'verse' else 4):
                add(mix, t + h*BEAT/(2 if sec == 'verse' else 4), hat(sec == 'drop' and h == 2))
        if sec == 'drop' and beat == 3 and bar % 2: add(mix, t+BEAT*0.75, K*0.7)
    # bass
    if sec in ('verse', 'drop', 'outro'):
        f = roots[bar % 4]; n = int(BAR*SR); t = np.arange(n)/SR
        saw = lambda ff: 2*((t*ff) % 1)-1
        b = saw(f)+saw(f*1.007)+0.5*saw(f*2.003)
        gate = ((t % (BEAT/2)) < BEAT*0.42).astype(float) if sec == 'drop' else 1.0
        b = np.tanh(5*lp(b, 151, 0.012 if sec == 'verse' else 0.022))*gate
        add(bassbus, b0, b*0.35)
    # monitor beeps = melody
    if sec in ('intro', 'verse', 'drop', 'outro'):
        step = 2 if sec in ('intro', 'verse') else 1
        for i in range(0, 8, step):
            if sec == 'intro' and i % 4: continue
            tt = b0 + i*BEAT/2; add(mix, tt, beep(lead[(i+bar) % 8])); events['beep'].append(tt)
    if sec == 'flat':
        n = int(BAR*SR); tt = np.arange(n)/SR
        add(mix, b0, np.sin(2*np.pi*1000*tt)*0.15*np.minimum(1, (BAR-tt)*8))

# noisy riser into drop
r0, r1 = 8*BAR, 10*BAR; n = int((r1-r0)*SR)
nz = rng.standard_normal(n)*np.linspace(0, 0.25, n)
add(mix, r0, np.diff(np.concatenate([[0], nz]))*0.6)
mix += bassbus*sc
mix += lp(rng.standard_normal(N), 51, 0.02)*0.015   # tape hiss
mix = np.tanh(1.3*mix); mix /= np.abs(mix).max()*1.05
st = np.stack([mix, np.roll(mix, 220)*0.9+mix*0.1], 1)
with wave.open(OUT+'.wav', 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((st*32767).astype('<i2').tobytes())

# ---------------- VIDEO ----------------
W, H, FPS = 1280, 720, 24
FB = '/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-ExtraBold.ttf'
FR = '/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf'
MAG, CYA, ACID = (255, 40, 150), (40, 240, 255), (190, 255, 60)
LYR = [
 (0, 4, "PATIENT 0x96 // ADMITTED"), (4, 6, "pulse is rising"), (6, 8, "take a deep breath"),
 (8, 9, "hold it..."), (9, 10, "hold it..."), (10, 12, "BASS IN THE BLOODSTREAM"),
 (12, 14, "scan me slow"), (14, 16, "i'm all frequency"), (16, 17, "BP: CRITICAL"),
 (17, 18, "- - - FLATLINE - - -"), (18, 20, "DX: TERMINALLY GROOVY"),
]
kicks = np.array(events['kick']); snares = np.array(events['snare'])

def since(arr, t):
    a = arr[arr <= t]; return t-a[-1] if len(a) else 9.0

def ecg(p):  # one heartbeat shape, p in [0,1)
    g = lambda c, w, a: a*np.exp(-((p-c)/w)**2)
    return g(.18, .03, .12)-g(.30, .008, .15)+g(.32, .01, 1.0)-g(.345, .01, .3)+g(.55, .05, .25)

def heart_pts(scale, cx, cy, rot):
    s = np.linspace(0, 2*np.pi, 140)
    x = 16*np.sin(s)**3; y = -(13*np.cos(s)-5*np.cos(2*s)-2*np.cos(3*s)-np.cos(4*s))
    return x, y

bg = np.zeros((H, W, 3), np.float32)
yy = np.linspace(0, 1, H)[:, None]
bg[..., 0] = 18+14*yy; bg[..., 1] = 4; bg[..., 2] = 22+20*(1-yy)
bgimg = Image.fromarray(bg.astype(np.uint8)); d = ImageDraw.Draw(bgimg)
for x in range(0, W, 40): d.line([(x, 0), (x, H)], (40, 14, 48))
for y in range(0, H, 40): d.line([(0, y), (W, y)], (40, 14, 48))
BG = np.asarray(bgimg).astype(np.float32)
SCAN = (np.arange(H) % 3 == 0)[:, None, None]*0.22

def frame(fi):
    t = fi/FPS; bar = int(t//BAR); sec = section(min(bar, NBARS-1))
    fr = np.random.default_rng(fi)
    kp = math.exp(-since(kicks, t)*7); sp = math.exp(-since(snares, t)*10)
    im = Image.fromarray(BG.astype(np.uint8)); glow = Image.new('RGB', (W, H)); dg = ImageDraw.Draw(glow)
    d = ImageDraw.Draw(im)
    # wireframe heart, MRI slices
    cx, cy = W*0.5, H*0.42; sc = 9 + 2.2*kp + (0.4*math.sin(t*3) if sec == 'intro' else 0)
    if sec == 'flat': sc = 9*(1-0.3*((t % BAR)/BAR))
    rot = t*0.6
    for k in range(14):
        z = (k/13-0.5)*2; r = math.sqrt(max(0, 1-z*z))
        s = np.linspace(0, 2*np.pi, 90)
        x = 16*np.sin(s)**3*r; y = -(13*np.cos(s)-5*np.cos(2*s)-2*np.cos(3*s)-np.cos(4*s))*r
        xr = x*math.cos(rot); zz = x*math.sin(rot)
        pts = list(zip(cx+xr*sc, cy+(y+z*9*math.sin(rot*0.5))*sc*0.9 + zz*0.0))
        col = MAG if k % 2 else CYA
        c = tuple(int(v*(0.35+0.65*kp)) for v in col)
        dg.line(pts+[pts[0]], c, 3); d.line(pts+[pts[0]], c, 1)
    # scan bar sweeping
    sy = int(cy-150*1.0 + ((t*0.8) % 1)*300)
    dg.rectangle([cx-260, sy-2, cx+260, sy+2], ACID)
    # ECG trace
    ey = H*0.80; xs = np.arange(0, W, 3)
    if sec == 'flat':
        ys = np.full(len(xs), ey) + fr.normal(0, .6, len(xs))
        ecol = (255, 50, 50)
    else:
        ph = (t - (W-xs)/W*2*BEAT)/BEAT + 0.65
        ys = ey - ecg(ph % 1)*110*(0.6+0.6*(sec == 'drop'))
        ecol = CYA if sec != 'drop' else ACID
    pts = list(zip(xs, ys))
    dg.line(pts, ecol, 6); d.line(pts, (255, 255, 255), 2)
    # HUD
    f1 = ImageFont.truetype(FR, 18); f2 = ImageFont.truetype(FB, 46)
    bpm = 0 if sec == 'flat' else (96 + (bar*3 if sec == 'drop' else 0) + int(fr.integers(-1, 2)))
    hud = [f"DX//SCAN  v9.6", f"HR  {bpm:3d} bpm", f"SpO2 {98 - (bar % 3)}%",
           f"BP  {120+bar*4}/{80+bar*2}", f"T+  {int(t//60):02d}:{t%60:05.2f}"]
    for i, s in enumerate(hud): d.text((28, 24+i*24), s, MAG if i == 0 else CYA, f1)
    d.text((W-300, 24), "REC ●" if int(t*2) % 2 else "REC", (255, 60, 60), f1)
    d.text((W-300, 48), f"SLICE {int(t*24)%128:03d}/128", CYA, f1)
    d.text((W-300, 72), f"SECTION: {sec.upper()}", ACID, f1)
    # lyric
    for b0, b1, txt in LYR:
        if b0*BAR <= t < b1*BAR:
            lt = t-b0*BAR; nch = min(len(txt), int(lt*22)+1)
            shown = txt[:nch]
            fs = ImageFont.truetype(FB, 54 if txt.isupper() else 50)
            tw = d.textlength(shown, fs)
            ty = H*0.62
            dg.text(((W-tw)/2, ty), shown, MAG, fs)
            d.text(((W-tw)/2, ty), shown, (255, 235, 245), fs)
    g = glow.filter(ImageFilter.GaussianBlur(9))
    a = np.asarray(im).astype(np.float32) + np.asarray(g).astype(np.float32)*1.6
    # glitch
    gl = sp*0.9 + (0.35 if sec == 'drop' else 0)*kp + (0.6 if sec == 'flat' and fr.random() < .2 else 0)
    if gl > 0.15:
        sh = int(gl*18)
        a[..., 0] = np.roll(a[..., 0], sh, 1); a[..., 2] = np.roll(a[..., 2], -sh, 1)
        for _ in range(int(gl*8)):
            y0 = int(fr.integers(0, H-30)); h = int(fr.integers(4, 40))
            a[y0:y0+h] = np.roll(a[y0:y0+h], int(fr.integers(-80, 80)), 1)
    a += fr.normal(0, 10, (H, W, 1))       # grain
    a *= 1 - SCAN
    vig = 1 - 0.55*((np.linspace(-1, 1, W)[None]**2 + np.linspace(-1, 1, H)[:, None]**2)/2)
    a *= vig[..., None] * (1 + 0.25*kp)
    if t < 1.0: a *= t
    if t > DUR-1.5: a *= max(0, (DUR-t)/1.5)
    return np.clip(a, 0, 255).astype(np.uint8).tobytes()

if __name__ == '__main__':
    nf = int(DUR*FPS)
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-i', OUT+'.wav', '-c:v', 'libx264', '-preset', 'medium',
        '-crf', '20', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-shortest', OUT+'.mp4'],
        stdin=subprocess.PIPE)
    with Pool(4) as p:
        for i, b in enumerate(p.imap(frame, range(nf), chunksize=8)):
            ff.stdin.write(b)
    ff.stdin.close(); ff.wait(); print('done', nf, 'frames')
