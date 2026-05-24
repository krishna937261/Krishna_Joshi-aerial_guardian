import cv2, os, glob, sys

seq_dir = sys.argv[1]   # path to img1 folder
out_path = sys.argv[2]  # output .mp4 path

imgs = sorted(glob.glob(os.path.join(seq_dir, "*.jpg")))
frame = cv2.imread(imgs[0])
h, w = frame.shape[:2]

out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
for p in imgs:
    out.write(cv2.imread(p))
out.release()
print(f"Saved {len(imgs)} frames to {out_path}")