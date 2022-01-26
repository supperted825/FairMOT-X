from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

from tracker import matching
from tracking_utils.kalman_filter import KalmanFilter
from tracking_utils.utils import *
from tracking_utils.log import logger

from utils.utils import map_to_orig_coords

from tracker.basetrack import BaseTrack, MCBaseTrack, TrackState
from models.model import create_model, load_model
from models.networks.yolox.utils.boxes import postprocess


class MCTrack(MCBaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, temp_feat, num_classes, cls_id, buff_size=30):
        """
        :param tlwh:
        :param score:
        :param temp_feat:
        :param num_classes:
        :param cls_id:
        :param buff_size:
        """
        # object class id
        self.cls_id = cls_id

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.track_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)

        # buffered features
        self.features = deque([], maxlen=buff_size)

        # fusion factor
        self.alpha = 0.9

    def update_features(self, feat):
        # L2 normalizing
        feat /= np.linalg.norm(feat)

        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1.0 - self.alpha) * feat

        self.features.append(feat)

        # L2 normalizing
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(tracks):
        if len(tracks) > 0:
            multi_mean = np.asarray([track.mean.copy() for track in tracks])
            multi_covariance = np.asarray([track.covariance for track in tracks])

            for i, st in enumerate(tracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0

            multi_mean, multi_covariance = Track.shared_kalman.multi_predict(multi_mean, multi_covariance)

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                tracks[i].mean = mean
                tracks[i].covariance = cov

    def reset_track_id(self):
        self.reset_track_count(self.cls_id)

    def activate(self, kalman_filter, frame_id):
        """Start a new track"""
        self.kalman_filter = kalman_filter  # assign a filter to each track?

        # update track id for the object class
        self.track_id = self.next_id(self.cls_id)

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.track_len = 0
        self.state = TrackState.Tracked  # set flag 'tracked'

        # self.is_activated = True
        if frame_id == 1:  # to record the first frame's detection result
            self.is_activated = True

        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        # kalman update
        self.mean, self.covariance = self.kalman_filter.update(self.mean,
                                                               self.covariance,
                                                               self.tlwh_to_xyah(new_track.tlwh))

        # feature vector update
        self.update_features(new_track.curr_feat)

        self.track_len = 0
        self.frame_id = frame_id

        self.state = TrackState.Tracked  # set flag 'tracked'
        self.is_activated = True

        if new_id:  # update track id for the object class
            self.track_id = self.next_id(self.cls_id)

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: Track
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.track_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(self.mean,
                                                               self.covariance,
                                                               self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked  # set flag 'tracked'
        self.is_activated = True  # set flag 'activated'

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()

        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_({}-{})_({}-{})'.format(self.cls_id, self.track_id, self.start_frame, self.end_frame)


class Track(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, temp_feat, buff_size=30):
        """
        :param tlwh:
        :param score:
        :param temp_feat:
        :param buff_size:
        """

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.track_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buff_size)  # 指定了限制长度
        self.alpha = 0.9

    def update_features(self, feat):
        # L2 normalizing
        feat /= np.linalg.norm(feat)

        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat

        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(tracks):
        if len(tracks) > 0:
            multi_mean = np.asarray([track.mean.copy() for track in tracks])
            multi_covariance = np.asarray([track.covariance for track in tracks])

            for i, st in enumerate(tracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0

            multi_mean, multi_covariance = Track.shared_kalman.multi_predict(multi_mean, multi_covariance)

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                tracks[i].mean = mean
                tracks[i].covariance = cov

    def reset_track_id(self):
        self.reset_track_count()

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter  # assign a filter to each tracklet?

        # update the track id
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.track_len = 0
        self.state = TrackState.Tracked  # set flag 'tracked'

        # self.is_activated = True
        if frame_id == 1:  # to record the first frame's detection result
            self.is_activated = True

        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(self.mean,
                                                               self.covariance,
                                                               self.tlwh_to_xyah(new_track.tlwh))

        self.update_features(new_track.curr_feat)
        self.track_len = 0
        self.state = TrackState.Tracked  # set flag 'tracked'
        self.is_activated = True
        self.frame_id = frame_id

        if new_id:  # update the track id
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: Track
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.track_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(self.mean,
                                                               self.covariance,
                                                               self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked  # set flag 'tracked'
        self.is_activated = True  # set flag 'activated'

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()

        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()  # numpy中的.copy()是深拷贝
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class YOLOTracker(object):
    def __init__(self, opt):
        self.opt = opt

        # ----- Build Model for Detections & ReID Feature Map
        
        print('Creating model...')
        model = create_model(opt.arch, opt=opt)
        
        assert opt.load_model is not None, "No Model to Load for tracking!"

        self.model = load_model(model, opt.load_model)
        try:
            print("Detection Loss Weight: ", self.model.head.s_det)
            print("ID Loss Weight: ", self.model.head.s_id)
        except:
            print("Model does not use uncertainty loss.")

        # ----- Set Model to Device & Evaluation Mode
        
        device = opt.device
        self.model.to(device).eval()

        # ----- Prepare Tracking Data Structures

        self.tracked_tracks_dict = defaultdict(list)  # value type: list[Track]
        self.lost_tracks_dict = defaultdict(list)  # value type: list[Track]
        self.removed_tracks_dict = defaultdict(list)  # value type: list[Track]

        self.frame_id = 0

        # ----- Tracking Hyperparameters
        
        self.buffer_size = int(opt.track_buffer)
        self.max_time_lost = self.buffer_size

        # ----- Kalman Filter for Tracking
        
        self.kalman_filter = KalmanFilter()


    def reset(self):
        """
        :return:
        """
        # Reset Tracker Buffer
        self.tracked_tracks_dict = defaultdict(list)    # value type: list[Track]
        self.lost_tracks_dict = defaultdict(list)       # value type: list[Track]
        self.removed_tracks_dict = defaultdict(list)    # value type: list[Track]

        # Reset Frame ID
        self.frame_id = 0

        # Reset Kalman Filter
        self.kalman_filter = KalmanFilter()


    def update_tracking(self, img, img0):
        """
        Update tracking result of the frame
        :param img:
        :param img0:
        :return:
        """
        
        opt = self.opt
        
        # Increment Frame ID
        self.frame_id += 1

        # ----- Reset Track IDs for First Frame
        if self.frame_id == 1:
            MCTrack.init_count(opt.num_classes)

        # ----- Get Image Sizes
        net_h, net_w = img.shape[2:]
        orig_h, orig_w, _ = img0.shape  # H×W×C

        # ----- Data Structures for Current Frame
        activated_tracks_dict = defaultdict(list)
        refined_tracks_dict = defaultdict(list)
        lost_tracks_dict = defaultdict(list)
        removed_tracks_dict = defaultdict(list)
        output_tracks_dict = defaultdict(list)

        # ----- Perform both Detection & ReID Extraction
        with torch.no_grad():
            
            # ----- Forward Pass in Eval Mode Returns BBOX & ReID
            pred, reid_map = self.model.forward(img)

            # ---- Applies NMS and Returns bboxes
            pred = postprocess(pred, self.opt.num_classes,
                               conf_thre=opt.det_thre,
                               nms_thre=opt.nms_thre,
                               class_agnostic=True)

            # ----- Extract Detections & Map Assuming Batch Size 1
            dets = pred[0]
            reid_map = reid_map[0]

            if dets is None:
                print('[Warning]: No objects detected.')
                return output_tracks_dict

            # ----- Extract ReID Features for Each Detection
            
            b, c, h, w = img.shape  # Network Input Img Size
            id_vects_dict = defaultdict(list)
            
            for i, det in enumerate(dets):
                # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
                x1, y1, x2, y2, obj_conf, class_conf, cls_id = det

                # L2 Normalize Feature Map
                reid_map = F.normalize(reid_map, dim=1)
                reid_dim, h_id_map, w_id_map = reid_map.shape

                # Map Center Point from Net Image Scale to ReID Map Scale
                center_x = x1 + x2 / 2
                center_y = y1 + y2 / 2
                center_x *= float(w_id_map) / float(w)
                center_y *= float(h_id_map) / float(h)

                # convert to int64 for indexing
                center_x += 0.5  # round
                center_y += 0.5
                center_x = center_x.long()
                center_y = center_y.long()
                center_x.clamp_(0, w_id_map - 1)  # to avoid the object center out of reid feature map's range
                center_y.clamp_(0, h_id_map - 1)

                # Get reID Feature Vector
                id_feat_vect = reid_map[:, center_y, center_x]      # 128 x 1 x 1
                id_feat_vect = id_feat_vect.squeeze()               # 128
                id_feat_vect = id_feat_vect.cpu().numpy()
                id_vects_dict[int(cls_id)].append(id_feat_vect)     # Add feat vect to dict(key: cls_id)

            # # ----- Map Detections to Original Input Image Coordinates
            # dets = map_to_orig_coords(dets, net_w, net_h, orig_w, orig_h)


        # ----- Process Tracking for Each Object Class
        for cls_id in range(self.opt.num_classes):
            
            cls_inds = torch.where(dets[:, -1] == cls_id)
            cls_dets = dets[cls_inds]                           # n_objs × 6
            cls_id_features = id_vects_dict[cls_id]              # n_objs × 128

            cls_dets = cls_dets.detach().cpu().numpy()
            cls_id_features = np.array(cls_id_features)

            # ----- Instantiate Track for Each Detection, Feature Pair
            
            
            # tlwh, score, temp_feat, num_classes, cls_id, buff_size=30
            if len(cls_dets) > 0:
                cls_detections = [
                                MCTrack(
                                    MCTrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], feature, self.opt.num_classes, cls_id, 30
                                    ) for (tlbrs, feature) in zip(cls_dets[:, :5], cls_id_features)
                                ]
            else:
                cls_detections = []

            # ----- Add New Tracks from Current Frame to Tracked Tracks
            
            unconfirmed_dict = defaultdict(list)
            tracked_tracks_dict = defaultdict(list)
            
            for track in self.tracked_tracks_dict[cls_id]:
                if not track.is_activated:
                    unconfirmed_dict[cls_id].append(track)
                else:
                    tracked_tracks_dict[cls_id].append(track)

            # ----- Association 1: With Feature Embedding
            
            # Build Track Pool for Current Frame with both Tracked & lost Tracks
            track_pool_dict = defaultdict(list)
            track_pool_dict[cls_id] = join_tracks(tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id])

            # Predict Current Location with Kalman Filter
            Track.multi_predict(track_pool_dict[cls_id])
            
            # Perform Embedding Distance Calculations & Assignment
            dists = matching.embedding_distance(track_pool_dict[cls_id], cls_detections)
            dists = matching.fuse_motion(self.kalman_filter, dists, track_pool_dict[cls_id], cls_detections)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)
            
            # Process Matched Pairs between Track Pool & Current Detections
            for i_tracked, i_det in matches:
                track = track_pool_dict[cls_id][i_tracked]
                det = cls_detections[i_det]
                # Update or Activate Matched Tracks Depending on State
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_tracks_dict[cls_id].append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refined_tracks_dict[cls_id].append(track)

            # ----- Association 2: With bbox IoU
            
            # Prepare Track Pool & Unmatched Detections from Current Frame
            cls_detections = [cls_detections[i] for i in u_detection]
            r_tracked_tracks = [track_pool_dict[cls_id][i] for i in u_track if track_pool_dict[cls_id][i].state == TrackState.Tracked]
            
            # Perform IoU Distance Calculations & Assignment
            dists = matching.iou_distance(r_tracked_tracks, cls_detections)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)
            
            # Process Matched Pairs between Track Pool & Current Detections
            for i_tracked, i_det in matches:
                track = r_tracked_tracks[i_tracked]
                det = cls_detections[i_det]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_tracks_dict[cls_id].append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refined_tracks_dict[cls_id].append(track)
                    
            # ----- Handle Untracked Tracks & Detections

            # Mark Remaining Tracks as Unmatched
            for it in u_track:
                track = r_tracked_tracks[it]
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_tracks_dict[cls_id].append(track)

            # Match Dormant Tracks with 2-Round Unmatched Dets
            cls_detections = [cls_detections[i] for i in u_detection]
            dists = matching.iou_distance(unconfirmed_dict[cls_id], cls_detections)
            matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.5)
            
            # Update Matched Tracks with New Detections
            for i_tracked, i_det in matches:
                unconfirmed_dict[cls_id][i_tracked].update(cls_detections[i_det], self.frame_id)
                activated_tracks_dict[cls_id].append(unconfirmed_dict[cls_id][i_tracked])

            # Remove Unmatched Tracks
            for it in u_unconfirmed:
                track = unconfirmed_dict[cls_id][it]
                track.mark_removed()
                removed_tracks_dict[cls_id].append(track)


            # ----- Init New Tracks with Final Unmatched Detections
            
            for i_new in u_detection:
                track = cls_detections[i_new]
                if track.score < opt.det_thre:
                    continue

                # Tracked But Not Activated
                track.activate(self.kalman_filter, self.frame_id)   # Note: Activate does not set 'is_activated' to be True
                activated_tracks_dict[cls_id].append(track)         # activated_tracks_dict may contain track with 'is_activated' False

            # ----- Update States
            for track in self.lost_tracks_dict[cls_id]:
                if self.frame_id - track.end_frame > self.max_time_lost:
                    track.mark_removed()
                    removed_tracks_dict[cls_id].append(track)

            # Update Tracked TracksL Add Activated & Refined Tracks
            self.tracked_tracks_dict[cls_id] = [t for t in self.tracked_tracks_dict[cls_id] if t.state == TrackState.Tracked]
            self.tracked_tracks_dict[cls_id] = join_tracks(self.tracked_tracks_dict[cls_id], activated_tracks_dict[cls_id])
            self.tracked_tracks_dict[cls_id] = join_tracks(self.tracked_tracks_dict[cls_id], refined_tracks_dict[cls_id])
            
            # Update Lost Tracks: Remove Tracked Tracks & Add New Lost Tracks & Remove Expired Tracks
            self.lost_tracks_dict[cls_id] = sub_tracks(self.lost_tracks_dict[cls_id], self.tracked_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id].extend(lost_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id] = sub_tracks(self.lost_tracks_dict[cls_id], self.removed_tracks_dict[cls_id])

            # Update Removed Tracks
            self.removed_tracks_dict[cls_id].extend(removed_tracks_dict[cls_id])

            # Conflict Resolution for Duplicated Tracks
            self.tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id] = remove_duplicate_tracks(self.tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id])

            # Get Scores of Lost Tracks
            output_tracks_dict[cls_id] = [track for track in self.tracked_tracks_dict[cls_id] if track.is_activated]

        return output_tracks_dict


def join_tracks(tracks_a, tracks_b):
    """
    join two track lists
    :param tracks_a:
    :param tracks_b:
    :return:
    """
    exists = {}
    join_tr_list = []

    for t in tracks_a:
        exists[t.track_id] = 1
        join_tr_list.append(t)

    for t in tracks_b:
        tr_id = t.track_id
        if not exists.get(tr_id, 0):
            exists[tr_id] = 1
            join_tr_list.append(t)

    return join_tr_list


def sub_tracks(tracks_a, tracks_b):
    tracks = {}

    for t in tracks_a:
        tracks[t.track_id] = t
    for t in tracks_b:
        tr_id = t.track_id
        if tracks.get(tr_id, 0):
            del tracks[tr_id]

    return list(tracks.values())


def remove_duplicate_tracks(tracks_a, tracks_b):
    p_dist = matching.iou_distance(tracks_a, tracks_b)
    pairs = np.where(p_dist < 0.15)
    dup_a, dup_b = list(), list()

    for a, b in zip(*pairs):
        time_a = tracks_a[a].frame_id - tracks_a[a].start_frame
        time_b = tracks_b[b].frame_id - tracks_b[b].start_frame
        if time_a > time_b:
            dup_b.append(b)  # choose short record time as duplicate
        else:
            dup_a.append(a)

    res_a = [t for i, t in enumerate(tracks_a) if not i in dup_a]
    res_b = [t for i, t in enumerate(tracks_b) if not i in dup_b]

    return res_a, res_b