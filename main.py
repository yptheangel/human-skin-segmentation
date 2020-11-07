import cv2
from model import PersonSegmentation

if __name__ == '__main__':
    # change 'cpu' to 'cuda' if you have pytorch cuda and your discrete GPU has enough VRAM
    # output size will autoscale to fit input image aspect ratio
    # if you want full image resolution set 'is_resize=False'
    ps = PersonSegmentation('cpu', is_resize=True, resize_size=480)
    filename = r"test_image.png"
    seg_map = ps.person_segment(filename)
    frame, frame_original = ps.decode_segmap(seg_map, filename)
    # skin_frame, skin2img_ratio = ps.skin_segment(frame)
    skin_frame, skin2img_ratio = ps.skin_segment_pro(frame)
    print(f"Skin to Image Percentage: {100 * skin2img_ratio:.2f}%")

    cv2.imshow("Original vs Person Seg vs Skin segmented", cv2.vconcat([frame_original, frame, skin_frame]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
