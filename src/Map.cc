/**
* This file is part of ORB-SLAM.
*
* Copyright (C) 2014 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <http://webdiis.unizar.es/~raulmur/orbslam/>
*
* ORB-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Map.h"

//needed for message publishing
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <ORB_SLAM/Keyframe_msg.h>
#include <ORB_SLAM/Landmark_msg.h>
#include <ORB_SLAM/BowWord.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/Point.h>
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"

namespace ORB_SLAM
{

Map::Map(ros::NodeHandle& node)
{
    mbMapUpdated= false;
    mnMaxKFid = 0;
    keyframe_pub = node.advertise<Keyframe_msg>("/orb_slam_keyframes",1);
}

void Map::AddKeyFrame(KeyFrame *pKF)
{
    boost::mutex::scoped_lock lock(mMutexMap);
    mspKeyFrames.insert(pKF);
    if(pKF->mnId>mnMaxKFid)
        mnMaxKFid=pKF->mnId;
    mbMapUpdated=true;

    //publish keyframe
    Keyframe_msg msg;
    cv_bridge::CvImage img_bridge;
    //sensor_msgs::Image img_msg;
    msg.frame_id = pKF->mnId;
    msg.header.stamp = ros::Time::now();
    cv::Mat pose = pKF->GetPoseInverse();
    cv::Mat projection = pKF->GetProjectionMatrix();
    for(int i=0;i<3;i++){
      for(int j=0;j<4;j++){
        msg.pose.push_back(pose.at<float>(i,j));
        msg.projection.push_back(projection.at<float>(i,j));
      }
    }

    cv::Mat descriptors = pKF->GetDescriptors().clone();
    //std::cout << "Number of descriptors" << descriptors.rows << "\n";
    for (int i=0;i<descriptors.rows;i++){
      for(int j=0;j<descriptors.cols;j++){
        msg.descriptors.push_back(descriptors.at<unsigned char>(i,j));
      }
    }
    msg.number_descriptors = descriptors.rows;
    msg.length_descriptors = descriptors.cols;
    //std::cout << "descriptors added " << msg.descriptors.size() <<"\n";


    std::vector<cv::KeyPoint> kpts = pKF->GetKeyPoints();
    for(int i=0;i<kpts.size();i++)
    {
      geometry_msgs::Point p;
      p.x = kpts[i].pt.x;
      p.y = kpts[i].pt.y;
      msg.keypoints.push_back(p);
    }

    std::vector<MapPoint*> mps = pKF->getMapPoints();
    if (mps.size()==0) return;
    for(int i=0;i<mps.size();i++)
    {
        Landmark_msg lm;
        cv::Mat pos = mps[i]->GetWorldPos();
        lm.pose.x=pos.at<float>(0);
        lm.pose.y=pos.at<float>(1);
        lm.pose.z=pos.at<float>(2);
        lm.id = mps[i]->mnId;
        cv::Mat d = mps[i]->GetDescriptor();
        for (int i =0;i<32;i++)
        {
          lm.descriptor.push_back(d.at<unsigned char>(i));
        }

        int index = mps[i]->GetIndexInKeyFrame(pKF);
        if(index>=0)
        {
          lm.index = index;
          msg.landmarks.push_back(lm);
        }
    }

  /*
  DBoW2::BowVector bow = pKF->GetBowVector();
  for(DBoW2::BowVector::const_iterator vit= bow.begin(), vend=bow.end(); vit!=vend; vit++)
    {
      BowWord bw;
      bw.word = vit->first;
      bw.value = vit->second;
      msg.bow_vector.push_back(bw);
    }
    */

    //std::cout << "Number of words added " << msg.bow_vector.size()<<"\n";
    std_msgs::Header header; // empty header
    header.stamp = ros::Time::now(); // time
    img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::MONO8, pKF->GetImage());
    img_bridge.toImageMsg(msg.image);
    keyframe_pub.publish(msg);
}

void Map::AddMapPoint(MapPoint *pMP)
{
    boost::mutex::scoped_lock lock(mMutexMap);
    mspMapPoints.insert(pMP);
    mbMapUpdated=true;
}

void Map::EraseMapPoint(MapPoint *pMP)
{
    boost::mutex::scoped_lock lock(mMutexMap);
    mspMapPoints.erase(pMP);
    mbMapUpdated=true;
}

void Map::EraseKeyFrame(KeyFrame *pKF)
{
    boost::mutex::scoped_lock lock(mMutexMap);
    mspKeyFrames.erase(pKF);
    mbMapUpdated=true;
}

void Map::SetReferenceMapPoints(const vector<MapPoint *> &vpMPs)
{
    boost::mutex::scoped_lock lock(mMutexMap);
    mvpReferenceMapPoints = vpMPs;
    mbMapUpdated=true;
}

vector<KeyFrame*> Map::GetAllKeyFrames()
{
    boost::mutex::scoped_lock lock(mMutexMap);
    return vector<KeyFrame*>(mspKeyFrames.begin(),mspKeyFrames.end());
}

vector<MapPoint*> Map::GetAllMapPoints()
{
    boost::mutex::scoped_lock lock(mMutexMap);
    return vector<MapPoint*>(mspMapPoints.begin(),mspMapPoints.end());
}

int Map::MapPointsInMap()
{
    boost::mutex::scoped_lock lock(mMutexMap);
    return mspMapPoints.size();
}

int Map::KeyFramesInMap()
{
    boost::mutex::scoped_lock lock(mMutexMap);
    return mspKeyFrames.size();
}

vector<MapPoint*> Map::GetReferenceMapPoints()
{
    boost::mutex::scoped_lock lock(mMutexMap);
    return mvpReferenceMapPoints;
}

bool Map::isMapUpdated()
{
    boost::mutex::scoped_lock lock(mMutexMap);
    return mbMapUpdated;
}

void Map::SetFlagAfterBA()
{
    boost::mutex::scoped_lock lock(mMutexMap);
    mbMapUpdated=true;

}

void Map::ResetUpdated()
{
    boost::mutex::scoped_lock lock(mMutexMap);
    mbMapUpdated=false;
}

unsigned int Map::GetMaxKFid()
{
    boost::mutex::scoped_lock lock(mMutexMap);
    return mnMaxKFid;
}

void Map::clear()
{
    for(set<MapPoint*>::iterator sit=mspMapPoints.begin(), send=mspMapPoints.end(); sit!=send; sit++)
        delete *sit;

    for(set<KeyFrame*>::iterator sit=mspKeyFrames.begin(), send=mspKeyFrames.end(); sit!=send; sit++)
        delete *sit;

    mspMapPoints.clear();
    mspKeyFrames.clear();
    mnMaxKFid = 0;
    mvpReferenceMapPoints.clear();
}

} //namespace ORB_SLAM
