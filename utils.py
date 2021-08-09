# TODO port from original TA

#
# /**
#  * Finds the proximal/distal point of a fruit (with boundary b, in image img) by looking at the symmetry of
#  * the top/bottom part of the fruit. It finds the boundary point along the best line of symmetry (and farthest
#  * out on the tip end, if the line passes through more than one boundary point).
#  */
# int findTipBySymmetry(CAdvImage * img, CPixelList * b, bool proximal)
# {
#         // Copy boundary into vector to allow indexing -- need modular arithmetic to circle around boundary
#         vector<CPixel> boundary(b->begin(), b->end());
#
#         int ctrX = getMidXHorizontal(boundary, img->GetHeight() / 2);
#         CPixel centerOfTarget(ctrX, img->GetHeight() / 2);
#
#         int top = getBoundaryIndexByX(boundary, centerOfTarget.x, true);  // center top of the target
#
#         int farthestXLeft = findFarthestXFromCenter(boundary, centerOfTarget, top, false);
#         int farthestXRight = findFarthestXFromCenter(boundary, centerOfTarget, top, true);
#
#         bool symmetric = isSymmetric(boundary, centerOfTarget, img->GetWidth(), img->GetHeight());
#         int farthestYIndex = findFarthestY(boundary, proximal);
#
#         bool curved = false;
#         double slope = findSlopeForPivot(boundary, centerOfTarget, farthestYIndex, farthestXLeft, farthestXRight, proximal, symmetric, &curved);
#
#         //CPixel axisPivot(img->GetWidth() / 2, img->GetHeight() / 2);                                                             // The pivot point is set at the center of frame image originally.
#         CPixel axisPivot = findPivot(boundary, centerOfTarget, slope, farthestXLeft, farthestXRight, proximal, symmetric, curved); // Modified in 2015.
#
#         double startAngle = findStartAngle(slope);
#
#         int bestTip;
#         double minError;
#         bool minErrorSet = false;
#
#         for (double angle = startAngle; angle <= startAngle + 90; angle++)
#         {
#                 // Avoid special cases (vertical lines) that arise when angle is 0
#                 if (angle == 0) {
#                         angle = 0.01;
#                 }
#
#                 // New x-axis at angle, through axisPivot
#                 double xAxisSlope = tan(angle * pi / 180);
#                 double xAxisIntercept = -xAxisSlope * axisPivot.x + axisPivot.y;
#
#                 // Left-most and right-most (or vice versa) boundary points on new x-axis
#                 int xAxisBegin, xAxisEnd;
#                 bool foundBegin = false;
#                 for (int i = 0; i < boundary.size(); i++) {
#                         if (isPointOnLine(boundary[i], xAxisSlope, xAxisIntercept)) {
#                                 if (! foundBegin) {
#                                         xAxisBegin = xAxisEnd = i;
#                                         foundBegin = true;
#                                 } else {
#                                         if (boundary[i].x < boundary[xAxisBegin].x) {
#                                                 xAxisBegin = i;
#                                         }
#                                         if (boundary[i].x > boundary[xAxisEnd].x) {
#                                                 xAxisEnd = i;
#                                         }
#                                 }
#                         }
#                 }
#                 if (! foundBegin) {
#                         continue;       // Serious problem -- skip this angle
#                 }
#                 if (proximal) {
#                         int tmp = xAxisBegin;
#                         xAxisBegin = xAxisEnd;
#                         xAxisEnd = tmp;
#                 }
#
#                 CPixel xAxisMid(axisPivot.x, axisPivot.y);
#                 // New y-axis, perpendicular to new x-axis, through midpoint between xAxisBegin and xAxisEnd points
#                 double yAxisSlope = -1 / xAxisSlope;
#                 double yAxisIntercept = -yAxisSlope * xAxisMid.x + xAxisMid.y;
#
#                 // Boundary point along new y-axis, on tip end, farthest from midpoint between xAxisBegin and xAxisEnd
#                 int currTip;
#                 float maxDist = 0;
#                 bool foundFirstCurrTip = false;
#                 for (int i = xAxisBegin; i != xAxisEnd; i = (i + 1) % boundary.size()) {
#                         if (isPointOnLine(boundary[i], yAxisSlope, yAxisIntercept)) {
#                                 float dist = boundary[i].Distance(xAxisMid);
#                                 if (! foundFirstCurrTip || dist > maxDist) {
#                                         currTip = i;
#                                         maxDist = boundary[currTip].Distance(xAxisMid);
#                                         foundFirstCurrTip = true;
#                                 }
#                         }
#                 }
#                 if (! foundFirstCurrTip) {
#                         continue;       // Serious problem -- skip this angle
#                 }
#
#                 // Sum of error, where error is the distance between the midpoint of 2 opposite boundary points and
#                 // the intersection point between the new y-axis and a line through those boundary points. The sum is
#                 // taken over a series of parallel lines from the new x-axis to the tip end of the fruit.
#                 double currError = 0;
#                 int numInSum = 0;
#                 vector<vector<int> > lrPointsOnLines = pointsOnLines(boundary, xAxisSlope, xAxisIntercept,
#                         (proximal ? -1 : 1), xAxisBegin, xAxisEnd);
#                 for (int numInterceptIncrs = 0; numInterceptIncrs < lrPointsOnLines.size(); numInterceptIncrs++)
#                 {
#                         // Get leftmost and rightmost points on the current line
#                         if (lrPointsOnLines[numInterceptIncrs].size() == 0)
#                                 continue;
#                         double intercept = xAxisIntercept + (proximal ? -1 : 1) * numInterceptIncrs;
#                         int leftmost = lrPointsOnLines[numInterceptIncrs].at(0);
#                         int rightmost = lrPointsOnLines[numInterceptIncrs].at(1);
#
#                         // Midpoint of boundary points
# double midX = (boundary[leftmost].x + boundary[rightmost].x) / 2.0,
#                                 midY = (boundary[leftmost].y + boundary[rightmost].y) / 2.0;
#                         CPixel mid(long(midX + 0.5), long(midY + 0.5));         // rounding
#
#                         // Intersection point between new y-axis and line through boundary points
#                         double intersectX = (intercept - yAxisIntercept) / (yAxisSlope - xAxisSlope),
#                                 intersectY = yAxisSlope * intersectX + yAxisIntercept;
#                         CPixel intersect(long(intersectX + 0.5), long(intersectY + 0.5));       // rounding
#
#                         float dist = mid.Distance(intersect);
#                         currError += fabs(dist);
#                         numInSum++;
#                 }
#                 if (numInSum == 0) {
#                         continue;       // Serious problem -- skip this angle
#                 }
#                 currError /= numInSum;
#
#                 if (! minErrorSet || currError < minError) {
#                         minError = currError;
#                         bestTip = currTip;
#                         minErrorSet = true;
#                 }
#
#                 if (-1 < angle && angle < 1) {
#                         angle = 0;
#                 }
#
#         }
#         //*/
# if (minErrorSet)
#         {
#                 // Normal case
#                 return bestTip;
#         }
#         else
#         {
#                 // Error case -- somehow no bestTip was chosen. Pick the leftmost boundary point having the max/min y-value.
#                 if (proximal) {
#                         int minIndex = 0;
#                         for (int i = 1; i < boundary.size(); i++) {
#                                 if (boundary[i].y < boundary[minIndex].y ||
#                                         (boundary[i].y == boundary[minIndex].y && boundary[i].x < boundary[minIndex].x)) {
#                                                 minIndex = i;
#                                 }
#                         }
#                         return minIndex;
#                 } else {
#                         int maxIndex = 0;
#                         for (int i = 1; i < boundary.size(); i++) {
#                                 if (boundary[i].y > boundary[maxIndex].y ||
#                                         (boundary[i].y == boundary[maxIndex].y && boundary[i].x < boundary[maxIndex].x)) {
#                                                 maxIndex = i;
#                                 }
#                         }
#                         return maxIndex;
#                 }
#         }
# }

# /**
# * Returns the x-value of the center of a horizontal line at given y-value.
# */
# int getMidXHorizontal(vector<CPixel> boundary, int Y)
# {
#         int leftMostX = boundary[getBoundaryIndexByY(boundary, Y, true)].x;
#         int     rightMostX = boundary[getBoundaryIndexByY(boundary, Y, false)].x;
#         int centerX = (leftMostX + rightMostX) / 2;
#         return centerX;
# }
#
# /**
# * Finds the point that has either the smallest or largest x-value on the boundary.
# */
# int findFarthestXFromCenter(vector<CPixel> boundary, CPixel& centerOfTarget, int top, bool isRight)
# {
#         int farthestPointIndex = 0;
#         int maxDifference = 0;
#         if (isRight)
#         {
#                 for (int i = 0; i < top; ++i)
#                 {
#                         if (boundary[i].x - centerOfTarget.x > maxDifference)
#                         {
#                                 farthestPointIndex = i;
#                                 maxDifference = abs(boundary[i].x - centerOfTarget.x);
#                         }
#                 }
#         }
#         else
#         {
#                 for (int i = top; i < boundary.size(); ++i)
#                 {
#                         if (centerOfTarget.x - boundary[i].x > maxDifference)
#                         {
#                                 farthestPointIndex = i;
#                                 maxDifference = abs(boundary[i].x - centerOfTarget.x);
#                         }
#                 }
#         }
#
#         return farthestPointIndex;
# }
#
#
#
# /**
# * Finds the point that has either the smallest or largest y-value on the boundary.
# */
# int findFarthestY(vector<CPixel> boundary, bool proximal)
# {
#         int farthestY_x = 0;  //x-value of higher shoulder point
#         int farthestYIndex = 0;
#         int count = 0;
#         if (proximal)
#         {
#                 for (int i = 0; i < boundary.size(); ++i)  // find the minimum y-value (y-value of the highest point) in the boundary
#                 {
#                         if (boundary[i].y == 0)
#                         {
#                                 farthestY_x += boundary[i].x;
#                                 count++;
#                         }
#                 }
#
#                 if (count != 0)
#                 {
#                         farthestY_x /= count;
#                 }
#                 farthestYIndex = getBoundaryIndexByX(boundary, farthestY_x, true);
#         }
#         else
#         {
#                 int maxY = 0;
#                 for (int i = 0; i < boundary.size(); ++i)  // find the maximum y-value (y-value of the lowest point) in the boundary
#                 {
#                         if (boundary[i].y > maxY)
#                         {
#                                 maxY = boundary[i].y;
#                         }
#                 }
#                 for (int j = 0; j < boundary.size(); ++j)
#                 {
#                         if (boundary[j].y == maxY)
#                         {
#                                 farthestY_x += boundary[j].x;
#                                 count++;
#                         }
#                 }
#
#                 if (count != 0)
#                 {
#                         farthestY_x /= count;
#                 }
#                 farthestYIndex = getBoundaryIndexByX(boundary, farthestY_x, false);
#         }
#         return farthestYIndex;
# }
#
#
# * Finds the best slope value, which determines the degree of start angle and the position of axis pivot, based on the shape of the target.
# * The shape is divided
# * Start angle and axis pivot are two important concepts that are used in findTipBySymmetry() function to look for the best symmetry line.
# */
# double findSlopeForPivot(vector<CPixel> boundary, CPixel& centerOfTarget, int farthestYIndex, int farthestXLeftIndex, int farthestXRightIndex, bool proximal, bool symmetric, bool* curved)
# {
#         double slope = 0;
#
#         int farthestY_x = boundary[farthestYIndex].x;  // x-value of lowest or highest point depending on whether it is calculating proximal point or distal point
#
#         if (proximal)  // proximal point calculation
#         {
#                 double slopeOnLeft, slopeOnRight;
#                 CPixel p1_left = boundary[getBoundaryIndexByY(boundary, centerOfTarget.y / 3, true)];
#                 CPixel p2_left = boundary[getBoundaryIndexByY(boundary, centerOfTarget.y / 2, true)];
#                 slopeOnLeft = slopeBetweenTwoPoints(p1_left.x, p1_left.y, p2_left.x, p2_left.y);
#
#                 CPixel p1_right = boundary[getBoundaryIndexByY(boundary, centerOfTarget.y / 3, false)];
#                 CPixel p2_right = boundary[getBoundaryIndexByY(boundary, centerOfTarget.y / 2, false)];
#                 slopeOnRight = slopeBetweenTwoPoints(p1_right.x, p1_right.y, p2_right.x, p2_right.y);
#
#                 if (symmetric)  // the target is symmetric
#                 {
#                         int proximalLeftIndex = getBoundaryIndexByY(boundary, centerOfTarget.y / 4, true);  // leftmost point on the line where y = centerY / 4
#                         int proximalRightIndex = getBoundaryIndexByY(boundary, centerOfTarget.y / 4, false);  // rightmost point on the line where y = centerY / 4
#
#                         int leftIndex = getBoundaryIndexByX(boundary, (3 * boundary[proximalLeftIndex].x + boundary[proximalRightIndex].x) / 4, true);
#                         int rightIndex = getBoundaryIndexByX(boundary, (3 * boundary[proximalRightIndex].x + boundary[proximalLeftIndex].x) / 4, true);
#
#                         if (slopeOnLeft * slopeOnRight > 0)  // the fruit is curved
#                         {
#                                 *curved = true;
#                                 // to see if the proximal area is rather flat or pointed
#                                 int headDifference = abs((farthestY_x - boundary[proximalLeftIndex].x) - (boundary[proximalRightIndex].x - farthestY_x));
#                                 int tailDifference = abs((farthestY_x - p2_left.x) - (p2_right.x - farthestY_x));
#
#                                 int higherLineLength = p1_right.x - p1_left.x;
#                                 int centerLineLength = boundary[getBoundaryIndexByY(boundary, centerOfTarget.y, false)].x - boundary[getBoundaryIndexByY(boundary, centerOfTarget.y, true)].x;
#
#                                 if (headDifference < tailDifference && (double(centerLineLength) / double(higherLineLength) >= 1.3))  // the proximal area is pointed
#                                 {
#                                         int midX = (boundary[proximalLeftIndex].x + boundary[proximalRightIndex].x) / 2;
#                                         int midY = (boundary[proximalLeftIndex].y + boundary[proximalRightIndex].y) / 2;
#                                         slope = -1 / slopeBetweenTwoPoints(boundary[farthestYIndex].x, boundary[farthestYIndex].y, midX, midY);
#                                 }
#                                 else  // the proximal area is rather flat
#                                 {
#                                         slope = slopeBetweenTwoPoints(boundary[leftIndex].x, boundary[leftIndex].y, boundary[rightIndex].x, boundary[rightIndex].y);
#                                 }
#                         }
#                         else  // the target is round
#                         {
#                                 int topY = boundary[getBoundaryIndexByX(boundary, centerOfTarget.x, true)].y;
#                                 if (((boundary[leftIndex].x < centerOfTarget.x) && (boundary[rightIndex].x > centerOfTarget.x)) && ((boundary[leftIndex].y < topY) && (boundary[rightIndex].y < topY))) // if it is clearly indented
#                                 {
#                                         slope = slopeBetweenTwoPoints(boundary[leftIndex].x, boundary[leftIndex].y, boundary[rightIndex].x, boundary[rightIndex].y);
#                                 }
#                                 else  // if it is not clearly indented
#                                 {
#                                         *curved = true;
#                                         // to see if the proximal area is rather flat or pointed
#                                         if (double(boundary[farthestXRightIndex].x - boundary[farthestXLeftIndex].x) / double(boundary[proximalRightIndex].x - boundary[proximalLeftIndex].x) > 1.5)  // the proximal area is pointed
#                                         {
#                                                 slope = -1 / slopeBetweenTwoPoints(boundary[farthestYIndex].x, boundary[farthestYIndex].y, centerOfTarget.x, centerOfTarget.y);
#                                         }
#                                         else
#                                         {
#                                                 slope = slopeBetweenTwoPoints(boundary[leftIndex].x, boundary[leftIndex].y, boundary[rightIndex].x, boundary[rightIndex].y);
#                                         }
#                                 }
#                         }
#                 }
#                 else  // the target is not symmetric
#                 {
#                         if (slopeOnLeft * slopeOnRight > 0)  // the target shape is curved
#                         {
#                                 *curved = true;
#                                 if ((p1_right.x - farthestY_x) > (farthestY_x - p1_left.x))  // the highest point is on left
#                                 {
#                                         int tempPoint = getBoundaryIndexByX(boundary, (3 * p1_right.x + p1_left.x) / 4, true);
#                                         slope = slopeBetweenTwoPoints(boundary[tempPoint].x, boundary[tempPoint].y, boundary[farthestYIndex].x, boundary[farthestYIndex].y);
#                                         if ((-1 / slope) * slopeOnRight < 0)
#                                         {
#                                                 slope = -1 / slope;
#                                         }
#                                 }
#                                 else  // the highest point is on right
#                                 {
#                                         int tempPoint = getBoundaryIndexByX(boundary, (3 * p1_left.x + p1_right.x) / 4, true);
#                                         slope = slopeBetweenTwoPoints(boundary[tempPoint].x, boundary[tempPoint].y, boundary[farthestYIndex].x, boundary[farthestYIndex].y);
#                                         if ((-1 / slope) * slopeOnLeft < 0)
#                                         {
#                                                 slope = -1 / slope;
#                                         }
#                                 }
#                         }
#                         else // the target shape is round
#                         {
#                                 *curved = false;
#                                 int leftIndex = getBoundaryIndexByX(boundary, (3 * p1_left.x + p1_right.x) / 4, true);
#                                 int rightIndex = getBoundaryIndexByX(boundary, (3 * p1_right.x + p1_left.x) / 4, true);
#                                 slope = slopeBetweenTwoPoints(boundary[leftIndex].x, boundary[leftIndex].y, boundary[rightIndex].x, boundary[rightIndex].y);
#                         }
#                 }
#         }
# else  // distal point calculation
#         {
#                 double slopeOnLeft, slopeOnRight;
#                 CPixel p1_left = boundary[getBoundaryIndexByY(boundary, centerOfTarget.y / 3 * 5, true)];
#                 CPixel p2_left = boundary[getBoundaryIndexByY(boundary, centerOfTarget.y / 2 * 3, true)];
#                 slopeOnLeft = slopeBetweenTwoPoints(p1_left.x, p1_left.y, p2_left.x, p2_left.y);
#
#                 CPixel p1_right = boundary[getBoundaryIndexByY(boundary, centerOfTarget.y / 3 * 5, false)];
#                 CPixel p2_right = boundary[getBoundaryIndexByY(boundary, centerOfTarget.y / 2 * 3, false)];
#                 slopeOnRight = slopeBetweenTwoPoints(p1_right.x, p1_right.y, p2_right.x, p2_right.y);
#
#                 if (symmetric)  // the target is symmetric
#                 {
#                         int distalLeftIndex = getBoundaryIndexByY(boundary, centerOfTarget.y / 4 * 7, true);  // leftmost point on the line where y = centerY / 4 * 7
#                         int distalRightIndex = getBoundaryIndexByY(boundary, centerOfTarget.y / 4 * 7, false);  // rightmost point on the line where y = centerY / 4 * 7
#
#                         // to see if the lower half of the target is curved or round
#                         if (slopeOnLeft * slopeOnRight > 0)  // the target is curved
#                         {
#                                 *curved = true;
#                                 if ((p1_right.x - farthestY_x) > (farthestY_x - p1_left.x))  // the lowest point is on left
#                                 {
#                                         int tempPoint = getBoundaryIndexByX(boundary, (3 * p1_right.x + p1_left.x) / 4, false);
#                                         slope = slopeBetweenTwoPoints(boundary[tempPoint].x, boundary[tempPoint].y, boundary[farthestYIndex].x, boundary[farthestYIndex].y);
#                                         if ((-1 / slope) * slopeOnRight < 0)
#                                         {
#                                                 slope = -1 / slope;
#                                         }
#                                 }
#                                 else  // the lowest point is on right
#                                 {
#                                         int tempPoint = getBoundaryIndexByX(boundary, (3 * p1_left.x + p1_right.x) / 4, false);
#                                         slope = slopeBetweenTwoPoints(boundary[tempPoint].x, boundary[tempPoint].y, boundary[farthestYIndex].x, boundary[farthestYIndex].y);
#                                         if ((-1 / slope) * slopeOnLeft < 0)
#                                         {
#                                                 slope = -1 / slope;
#                                         }
#                                 }
#                         }
#                         else  // the target is round
#                         {
#                                 int minY = boundary[farthestYIndex].y;
#                                 int minYIndex = farthestYIndex;
#                                 if (centerOfTarget.x / double(centerOfTarget.y) >= 1.1)  // the target shape is flat
#                                 {
#                                         int distanceFromCenter = abs(farthestY_x - centerOfTarget.x);
#                                         int leftIndex = getBoundaryIndexByX(boundary, centerOfTarget.x - distanceFromCenter, false);
#                                         int rightIndex = getBoundaryIndexByX(boundary, centerOfTarget.x + distanceFromCenter, false);
#                                         slope = slopeBetweenTwoPoints(boundary[leftIndex].x, boundary[leftIndex].y, boundary[rightIndex].x, boundary[rightIndex].y);
#                                 }
#                                 else
#                                 {
#                                         *curved = true;
#                                         if ((centerOfTarget.y / double(centerOfTarget.x) >= 1.8))  // the target shape is long
#                                         {
#                                                 int midX = (boundary[distalLeftIndex].x + boundary[distalRightIndex].x) / 2;
#                                                 int midY = (boundary[distalLeftIndex].y + boundary[distalRightIndex].y) / 2;
#                                                 slope = -1 / slopeBetweenTwoPoints(boundary[farthestYIndex].x, boundary[farthestYIndex].y, midX, midY);
#                                         }
#                                         else
#                                         {
#                                                 slope = -1 / slopeBetweenTwoPoints(centerOfTarget.x, centerOfTarget.y, boundary[farthestYIndex].x, boundary[farthestYIndex].y);
#                                         }
#                                 }
#                         }
#                 }
#                 else  // the target is not symmetric
#                 {
#                         if (slopeOnLeft * slopeOnRight > 0)  // the target shape is curved
#                         {
#                                 *curved = true;
#                                 if ((p1_right.x - farthestY_x) > (farthestY_x - p1_left.x))  // the lowest point is on left
#                                 {
#                                         int tempPoint = getBoundaryIndexByX(boundary, (3 * p1_right.x + p1_left.x) / 4, false);
#                                         slope = slopeBetweenTwoPoints(boundary[tempPoint].x, boundary[tempPoint].y, boundary[farthestYIndex].x, boundary[farthestYIndex].y);
#                                         if ((-1 / slope) * slopeOnRight < 0)
#                                         {
#                                                 slope = -1 / slope;
#                                         }
#                                 }
#                                 else  // the lowest point is on right
#                                 {
#                                         int tempPoint = getBoundaryIndexByX(boundary, (3 * p1_left.x + p1_right.x) / 4, false);
#                                         slope = slopeBetweenTwoPoints(boundary[tempPoint].x, boundary[tempPoint].y, boundary[farthestYIndex].x, boundary[farthestYIndex].y);
#                                         if ((-1 / slope) * slopeOnLeft < 0)
#                                         {
#                                                 slope = -1 / slope;
#                                         }
#                                 }
#                         }
#                         else // the target shape is round
#                         {
#                                 *curved = false;
#                                 slope = -1 / slopeBetweenTwoPoints(boundary[farthestYIndex].x, boundary[farthestYIndex].y, centerOfTarget.x, centerOfTarget.y);
#                         }
#                 }
#         }
#         return slope;
# }

# CPixel findPivot(vector < CPixel > boundary, CPixel & centerOfTarget, double slope, int farthestXLeftIndex, int farthestXRightIndex, bool proximal, bool symmetric, bool curved)
# {
#         int
# tempX, tempY; // temporary
# x - value and y - value
# of
# axisPivot
# if (symmetric) // the
# target is symmetric
# {
# if (proximal) // proximal
# point
# calculation
# {
# if ((boundary[farthestXLeftIndex].y - centerOfTarget.y) > 100 | | (boundary[farthestXRightIndex].y - centerOfTarget.y) > 100) //
# if the lower half of the target is bigger than the upper half
# {
# tempY = centerOfTarget.y / 4;
# }
# else
# {
# if (curved) // the target shape is curved
# {
# tempY = centerOfTarget.y / 4;
# }
# else // the target shape is round
# {
# tempY = (boundary[farthestXLeftIndex].y + boundary[farthestXRightIndex].y) / 2;
# }
# }
# }
# else // distal point calculation
# {
# if (curved) // the target shape is curved
# {
# tempY = centerOfTarget.y / 4 * 7;
# }
# else // the target shape is round
# {
# tempY = centerOfTarget.y / 5 * 6;
# }
# }
# }
# else // the target is not symmetric
# {
# if (proximal) // proximal point calculation
# {
# if (curved) // the target shape is curved
# {
# tempY = centerOfTarget.y / 4;
# }
# else // the target shape is round
# {
# tempY = centerOfTarget.y / 3;
# }
# }
# else // distal point calculation
# {
# if (curved) // the target shape is curved
# {
# tempY = centerOfTarget.y / 4 * 7;
# }
# else // the target shape is round
# {
# tempY = centerOfTarget.y / 3 * 5;
# }
# }
# }
# tempX = getMidXHorizontal(boundary, tempY);
#
# // Calculates the best pivot point.
# // Using the slope value and y-value, it sets up a x-axis, finds the endpoints on the x-axis, and takes the x-value of the midpoint as the x-value of the pivot point.
# double intercept = -slope * tempX + tempY;
# int begin, end;
# int count = 0;
# vector < int > results;
# bool foundBegin = false;
# for (int i = 0; i < boundary.size(); i++) {
# if (isPointOnLine(boundary[i], slope, intercept)) {
# if (!foundBegin) {
# begin = end = i;
# results.push_back(i);
# count++;
# foundBegin = true;
# }
# else {
# if (boundary[i].x < boundary[begin].x) {
# if (abs(i - results[count - 1]) > 5)
# {
# results.push_back(i);
# count++;
# }
# begin = i;
# }
# if (boundary[i].x > boundary[end].x) {
# end = i;
# }
# }
# }
# }
#
# if (count > 1)
# {
# begin = results[1];
# }
#
# CPixel axisPivot((boundary[begin].x + boundary[end].x) / 2,
# (boundary[begin].y + boundary[end].y) / 2);
# return axisPivot;
# }


# double findStartAngle(double slope)
# {
#         double newSlope = 0;
#         double startAngle = 0;
#         if (-slope > 0)
#         {
#                 newSlope = tan(atan(-1 / -slope) - (45 * pi / 180));
#                 if (newSlope < 0)
#                 {
#                         startAngle = -1 * (180 - (atan(newSlope) * -180 / pi));
#                 }
#                 else
#                 {
#                         startAngle = -1 * atan(newSlope) * 180 / pi;
#                 }
#         }
#         else
#         {
#                 newSlope = tan(atan(-1 / -slope) - (45 * pi / 180));
#                 if (newSlope < 0)
#                 {
#                         startAngle = -1 * (atan(newSlope) * 180 / pi);
#                 }
#                 else
#                 {
#                         startAngle = -1 * atan(newSlope) * 180 / pi;
#                 }
#         }
#         return startAngle;
# }
#
#
#
# /**
# * Returns true if the target is symmetric, false otherwise.
# * The distance between the center of target and the center of image is used as the criterion to decide
# * whether a target is symmetric or not.
# */
# bool isSymmetric(vector<CPixel> boundary, CPixel& centerOfTarget, int width, int height)
# {
#         if (abs(width / 2 - centerOfTarget.x) > 50)
#         {
#                 return false;
#         }
#         else
#         {
#                 return true;
#         }
# }
#
#
# /**
# * Returns the index of the leftmost or rightmost point on the line at given y-value
# */
# int getBoundaryIndexByY(vector<CPixel> boundary, int Y, bool isLeft)
# {
#         if (isLeft)
#         {
#                 int leftIndex = 0;
#                 int minX = 999999;
#                 for (int i = 0; i < boundary.size(); ++i)
#                 {
#                         if (boundary[i].y == Y)
#                         {
#                                 if (boundary[i].x < minX)
#                                 {
#                                         minX = boundary[i].x;
#                                         leftIndex = i;
#                                 }
#                         }
#                 }
#                 return leftIndex;
#         }
#         else
#         {
#                 int rightIndex = 0;
#                 int maxX = 0;
#                 for (int i = 0; i < boundary.size(); ++i)
#                 {
#                         if (boundary[i].y == Y)
#                         {
#                                 if (boundary[i].x > maxX)
#                                 {
#                                         maxX = boundary[i].x;
#                                         rightIndex = i;
#                                 }
#                         }
#                 }
#                 return rightIndex;
#         }
# }
#
# /**
# * Returns the index of the top or bottom point on the line at given x-value
# */
# int getBoundaryIndexByX(vector<CPixel> boundary, int X, bool isTop)
# {
#         if (isTop)
#         {
#                 int topIndex = 0;
#                 int minY = 999999;
#                 for (int i = 0; i < boundary.size(); ++i)
#                 {
#                         if (boundary[i].x == X)
#                         {
#                                 if (boundary[i].y < minY)
#                                 {
#                                         minY = boundary[i].y;
#                                         topIndex = i;
#                                 }
#                         }
#                 }
#                 return topIndex;
#         }
#         else
#         {
#                 int bottomIndex = 0;
#                 int maxY = 0;
#                 for (int i = 0; i < boundary.size(); ++i)
#                 {
#                         if (boundary[i].x == X)
#                         {
#                                 if (boundary[i].y > maxY)
#                                 {
#                                         maxY = boundary[i].y;
#                                         bottomIndex = i;
#                                 }
#                         }
#                 }
#                 return bottomIndex;
#         }
# }
#
#
# /**
# * Calculates and returns the slope between point (x1, y1) and point (x2, y2)
# */
# double slopeBetweenTwoPoints(int x1, int y1, int x2, int y2)
# {
#         if (abs(x1 - x2) == 0)  // error handling, avoid division by zero error.
#         {
#                 return 999.9;
#         }
#         else
#         {
#                 return double(y1 - y2) / double(x1 - x2);
#         }
# }

