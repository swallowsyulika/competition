import numpy as np
arr = np.array([[
                    632,
                    522
                ],
                [
                    773,
                    513
                ],
                [
                    785,
                    624
                ],
                [
                    636,
                    627
                ]])


# def order_points(pts):
#     # initialzie a list of coordinates that will be ordered
#     # such that the first entry in the list is the top-left,
#     # the second entry is the top-right, the third is the
#     # bottom-right, and the fourth is the bottom-left
#     rect = np.zeros((4, 2), dtype="float32")
#     # the top-left point will have the smallest sum, whereas
#     # the bottom-right point will have the largest sum
#     s = pts.sum(axis=1)
    
#     rect[0] = pts[np.argmin(s)]
#     rect[2] = pts[np.argmax(s)]
#     # now, compute the difference between the points, the
#     # top-right point will have the smallest difference,
#     # whereas the bottom-left will have the largest difference
#     diff = np.diff(pts, axis=1)
#     rect[1] = pts[np.argmin(diff)]
#     rect[3] = pts[np.argmax(diff)]
#     # return the ordered coordinates
#     return rect
def order_points(pts):
	rect = np.zeros((4, 2), dtype="float32")
	
	indicies = list(range(4))
	
	s = pts.sum(axis=1)
	top_left_idx = np.argmin(s)	
	indicies.remove(top_left_idx)
	
	pts_clone = np.array(pts)
	r_candidate_idx = np.argmax(pts[:, 0])
	indicies.remove(r_candidate_idx)
	r_candidate = pts[r_candidate_idx]
	
	pts_clone[r_candidate_idx][0] = -1
	
	r_candidate2_idx = np.argmax(pts_clone[:, 0])
	r_candidate2 = pts[r_candidate2_idx]
	indicies.remove(r_candidate2_idx)

	if r_candidate[1] > r_candidate2[1]:
		top_right_idx = r_candidate_idx
		bottom_right_idx = r_candidate2_idx
	else:			
		top_right_idx = r_candidate2_idx
		bottom_right_idx = r_candidate_idx
	
	bottom_left_idx = indicies[0]
	
	if len(indicies) != 1:
		print("[!] Something went wrong!!!")

	rect[0] = pts[top_left_idx]
	rect[1] = pts[top_right_idx]
	rect[2] = pts[bottom_right_idx]
	rect[3] = pts[bottom_left_idx]

	return rect
	
print(order_points(arr))