import cv2

if __name__ == '__main__':
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 500)
    fontScale = 1
    fontColor = (0, 0, 0)
    lineType = 2
    video = cv2.VideoCapture('video-samples/Baxter_TD3_cooperative_episode-3628.avi')
    i = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frame_conv = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(frame_conv,
                    'Episode: 3628, Step: {}'.format(str(i+1)),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        cv2.imwrite('video-stills/still{}.png'.format(str(i)), frame_conv)
        i += 1
    video.release()
    cv2.destroyAllWindows()
