function iou=bb_intersection_over_union(boxA, boxB)
%boxA and boxB specify the upper left corner and size 
%of that corresponding bounding box in pixels [x y width height]. The iou 
%is returned.
%source:https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
%determina the (x, y)-coordinates of the intersection rectangle
xa=max(boxA(1),boxB(1));
ya=max(boxA(2),boxB(2));
xb=min(boxA(1)+boxA(3),boxB(1)+boxB(3));
yb=min(boxA(2)+boxA(4),boxB(2)+boxB(4));

%compute the area of intersection rectangle
interArea = double(max(0, xb - xa) * max(0, yb - ya));

% compute the area of both the prediction and ground-truth
% rectangles
boxAArea = double(boxA(3)*boxA(4));
boxBArea = double(boxB(3)*boxB(4));

% compute the intersection over union by taking the intersection
% area and dividing it by the sum of prediction + ground-truth
% areas - the interesection area
iou = interArea./(boxAArea + boxBArea - interArea);

end
