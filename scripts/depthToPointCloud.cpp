/*
 * TI Voxel Lib component.
 *
 * Copyright (c) 2014 Texas Instruments Inc.
 */

#include "PointCloudTransform.h"

#include <limits>
#include <algorithm>

#include "Logger.h"

#define _USE_MATH_DEFINES
#include <math.h>

Point &PointCloudTransform::getDirection(int row, int col)
{
  return directions[col * width + row];
}

PointCloudTransform::PointCloudTransform(uint32_t left, uint32_t top, uint32_t width, uint32_t height, 
                                         uint32_t rowsToMerge, uint32_t columnsToMerge,
                                         float fx, float fy, float cx, float cy,
                                         float k1, float k2, float k3, float p1, float p2)
{
  this->left = left;
  this->top = top;
  this->width = width;
  this->height = height;
  this->rowsToMerge = rowsToMerge;
  this->columnsToMerge = columnsToMerge;
  this->fx = fx;
  this->fy = fy;
  this->cx = cx;
  this->cy = cy;
  this->k1 = k1;
  this->k2 = k2;
  this->k3 = k3;
  this->p1 = p1;
  this->p2 = p2;
  directions.clear();
  _init();
}

// Private methods


Point PointCloudTransform::_lensCorrection(const Point &normalizedScreen)
{
  float x_ = (normalizedScreen.x - cx)/fx;
  float y_ = (normalizedScreen.y - cy)/fy;
  float r2 = x_ * x_ + y_ * y_;
  float r4 = r2 * r2;
  float r6 = r2 * r4;
  
  float x__ = x_ * (1.0f + k1 * r2 + k2 * r4 + k3 * r6) + 2.0f * p1 * x_ * y_ + p2 * (r2 + 2.0f * x_ * x_);
  float y__ = y_ * (1.0f + k1 * r2 + k2 * r4 + k3 * r6) + p1 * (r2 + 2.0f * y_ * y_) + 2.0f * p2 * x_ * y_;
  
  return (Point(x__, y__));
}


Point PointCloudTransform::_normalizedScreenToUnitWorld(const Point &normalizedScreen)
{
  float _norm = 1.0f / (float)sqrt(normalizedScreen.x * normalizedScreen.x
                                   + normalizedScreen.y * normalizedScreen.y
                                   + 1.0f);
  return Point(normalizedScreen.x * _norm, normalizedScreen.y * _norm, _norm);
}

Point PointCloudTransform::_screenToNormalizedScreen(const Point &screen, bool verify)
{
  int iters = 100;
  float xs, ys;
  float yss = ys = (screen.y - cy) / fy;
  float xss = xs = (screen.x - cx) / fx;

  for(int j = 0; j < iters; j++)
  {
    float r2 = xs * xs + ys * ys;
    float icdist = 1.0f / (1 + ((k3 * r2 + k2) * r2 + k1) * r2);
    float deltaX = 2 * p1 * xs * ys + p2 * (r2 + 2 * xs * xs);
    float deltaY = p1 * (r2 + 2 * ys * ys) + 2 * p2 * xs * ys;
    xs = (xss - deltaX)*icdist;
    ys = (yss - deltaY)*icdist;
  }

  if(verify)
  {
    float x_ = xs;
    float y_ = ys;
    float r2 = x_ * x_ + y_ * y_;
    float r4 = r2 * r2;
    float r6 = r2 * r4;

    float x__ = x_ * (1.0f + k1 * r2 + k2 * r4 + k3 * r6) + 2.0f * p1 * x_ * y_ + p2 * (r2 + 2.0f * x_ * x_);
    float y__ = y_ * (1.0f + k1 * r2 + k2 * r4 + k3 * r6) + p1 * (r2 + 2.0f * y_ * y_) + 2.0f * p2 * x_ * y_;

    if(fabs(x__ - xss) > FLOAT_EPSILON || fabs(y__ - yss) > FLOAT_EPSILON)
    {
      return Point(POINT_INVALID, 0);
    }
    else
    {
      return Point(xs, ys);
    }
  }
  else
  {
    return Point(xs, ys);
  }
}

void PointCloudTransform::_init()
{
  directions.clear();
  for(int v = top; v < top + height; v++)
  {
    for(int u = left; u < left + width; u++)
    {
      Point normalizedScreen = _screenToNormalizedScreen(Point(u, v), true);
      //Point normalizedScreen = _lensCorrection(Point(u, v));
      Point dir = _normalizedScreenToUnitWorld(normalizedScreen);
      directions.push_back(dir);
    }
  }
}

bool PointCloudTransform::depthToPointCloud(const Vector<float> &distances, PointCloudFrame &pointCloudFrame)
{
  uint w = (width + columnsToMerge - 1)/columnsToMerge, h = (height + rowsToMerge - 1)/rowsToMerge;

  if(distances.size() < w*h ||
    pointCloudFrame.size() < w*h)
    return false;

  for(int v = 0; v < height; v += rowsToMerge)
  {
    for(int u = 0; u < width; u += columnsToMerge)
    {
      int idx = v * width + u;
      int idx2 = v/rowsToMerge * width/columnsToMerge + u/columnsToMerge;
      Point *p = pointCloudFrame[idx2];

      if(p)
        *p = directions[idx] * distances[idx2];
//       else
//       {
//         logger(LOG_ERROR) << "PointCloudTransform: Could not set point at (" << u << ", " << v << ")" << std::endl;
//         return false;
//       }
    }
  }
  return true;
}
