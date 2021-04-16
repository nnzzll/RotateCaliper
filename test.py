import cv2
import matplotlib.pyplot as plt
from func import RotatingCaliper

image = cv2.imread('test.png',0)
contour,_ = cv2.findContours(image,0,1)
contour = contour[0].reshape(-1,2)

max_dist,min_dist,max_res,min_res = RotatingCaliper(contour)
p1,p2 = max_res
p3,p4 = min_res

plt.imshow(image,cmap='gray')
plt.plot(contour[:,0],contour[:,1],'-',c='cyan')
plt.plot([p1.x,p2.x],[p1.y,p2.y],'o',c='r')
plt.plot([p1.x,p2.x],[p1.y,p2.y],'-',c='r')
plt.plot([p3.x,p4.x],[p3.y,p4.y],'o',c='g')
plt.plot([p3.x,p4.x],[p3.y,p4.y],'-',c='g')
plt.show()