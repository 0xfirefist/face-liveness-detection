import cv2

def save_image(filename, frame):
    cv2.imwrite(filename, frame)

cap = cv2.VideoCapture(0)

image_index = 0
while True:
    if image_index == 221:
        break
    # Capture frame-by-frame
    ret, frame = cap.read()

    # # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',frame)

    # save frame
    if image_index > 200:
        save_image("test/real/" + str(image_index-200+80) + ".jpg", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if ret == False:
        break

    image_index += 1


cap.release()
cv2.destroyAllWindows()
