import cv2, numpy as np, matplotlib.pyplot as plt, os

# ------------------------------------------------------------ #
#  Yardımcı fonksiyonlar – hepsi zincir halinde                #
# ------------------------------------------------------------ #
def gray_world(img):
    f      = img.astype(np.float32)
    gain   = f.mean() / (f.mean(axis=(0,1)) + 1e-6)
    return np.clip(f * gain, 0, 255).astype(np.uint8)

def msr(img, sig=(15, 80, 250)):
    def ssr(ch, s):
        return cv2.normalize(cv2.log(ch+1.) - cv2.log(cv2.GaussianBlur(ch,(0,0),s)+1.),
                             None, 0, 255, cv2.NORM_MINMAX)
    b,g,r = cv2.split(img)
    merge = lambda c: sum(ssr(c,s) for s in sig) / len(sig)
    return cv2.merge([merge(b).astype(np.uint8),
                      merge(g).astype(np.uint8),
                      merge(r).astype(np.uint8)])

def homomorphic(gray, cutoff=30):
    r,c = gray.shape
    dft = cv2.dft(gray.astype(np.float32)/255., flags=cv2.DFT_COMPLEX_OUTPUT)
    dft = np.fft.fftshift(dft)
    u,v = np.meshgrid(np.arange(-c//2,c//2), np.arange(-r//2,r//2))
    H   = 1 - 1/(1 + (np.sqrt(u**2+v**2)/cutoff)**4)
    H   = H[...,None].repeat(2,2)
    dft *= H
    f   = cv2.idft(np.fft.ifftshift(dft))
    out = cv2.normalize(cv2.magnitude(f[...,0],f[...,1]), None, 0, 255, cv2.NORM_MINMAX)
    return out.astype(np.uint8)

def preprocess(img):
    k    = np.ones((3,3),np.uint8)
    gray = cv2.cvtColor(msr(gray_world(img)), cv2.COLOR_BGR2GRAY)
    homo = homomorphic(gray)
    cla  = cv2.createCLAHE(3,(8,8)).apply(homo)
    bw   = cv2.adaptiveThreshold(cla,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY,11,2)
    mask = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN , k, 1)
    return cv2.dilate(mask, k, 1)          # kalınlaştırılmış maske

# ------------------------------------------------------------ #
#  Ana akış                                                    #
# ------------------------------------------------------------ #
img_path = r"C:\kortcizim.png"            # görüntü yolu
orig     = cv2.imread(img_path)
if orig is None:
    raise FileNotFoundError(img_path)

mask  = preprocess(orig)

# Çizgi tespiti ve çizim
lsd   = cv2.createLineSegmentDetector(scale=0.7)
lines, _, _, _ = lsd.detect(mask)

drawn = orig.copy()
if lines is not None:
    for x1,y1,x2,y2 in np.squeeze(lines):
        cv2.line(drawn, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)

# ------------------------------------------------------------ #
#  Göster – hiçbir şey kaydetmiyoruz                           #
# ------------------------------------------------------------ #
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.title("Court Mask")
plt.imshow(mask, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Court Lines Drawn")
plt.imshow(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()
