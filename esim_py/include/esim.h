#pragma once

#include <vector>

#include <boost/filesystem.hpp>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>


struct Event
{
  Event(int x, int y, double t, int polarity)
  : x_(x), y_(y), t_(t), polarity_(polarity)
  {}

  bool operator<(Event& other)
  {
    return t_ < other.t_;
  }

  int x_, y_;
  double t_;
  int polarity_;
};

/*
 * The EventSimulator takes as input a sequence of stamped images,
 * assumed to be sampled at a "sufficiently high" framerate,
 * and simulates the principle of operation of an idea event camera
 * with a constant contrast threshold C.
 * Pixel-wise intensity values are linearly interpolated in time.
 *
 * The pixel-wise voltages are reset with the values from the first image
 * which is passed to the simulator.
 */
class EventSimulator
{
public:
  EventSimulator(float contrast_threshold_pos, 
                 float contrast_threshold_neg, 
                 float refractory_period, 
                 float log_eps,
                 bool use_log_img);

  Eigen::MatrixXd generateFromFolder(std::string image_folder, std::string timestamps_file_path);
  Eigen::MatrixXd generateFromVideo(std::string video_path, std::string timestamps_file_path);
  Eigen::MatrixXd generateFromStampedImageSequence(std::vector<std::string> image_paths, std::vector<double> timestamps);

  void setParameters(float contrast_threshold_pos, 
                     float contrast_threshold_neg,
                     float refractory_period,
                     float log_eps,
                     bool use_log_img)
  {
    contrast_threshold_pos_ = contrast_threshold_pos;
    contrast_threshold_neg_ = contrast_threshold_neg;
    refractory_period_ = refractory_period;
    log_eps_ = log_eps;
    use_log_img_ = use_log_img;
  }

private:
      
  Eigen::MatrixXd vec_to_eigen_matrix(std::vector<Event>& events_vec)
  {
    Eigen::MatrixXd events(events_vec.size(), 4);
    for (int i=0; i<events_vec.size(); i++)
    {
        Event& event = events_vec[i]; 
        events(i,0) = event.x_;
        events(i,1) = event.y_;
        events(i,2) = event.t_;
        events(i,3) = event.polarity_;
    }
    return events;
  }

  void imageCallback(const cv::Mat& img, double time, std::vector<Event>& events);
  void init(const cv::Mat &img, double time);  
   
  void read_directory_from_path(const std::string& name, std::vector<std::string>& v)
  {
      boost::filesystem::path p(name);
      boost::filesystem::directory_iterator start(p);
      boost::filesystem::directory_iterator end;

      auto path_leaf_string = [](const boost::filesystem::directory_entry& entry) -> std::string {return entry.path().string();};

      std::transform(start, end, std::back_inserter(v), path_leaf_string);

      std::sort(v.begin(), v.end());
  }  

  float contrast_threshold_pos_; 
  float contrast_threshold_neg_;
  float refractory_period_;
  float log_eps_;
  bool use_log_img_;

  bool is_initialized_;
  double current_time_;
  cv::Mat ref_values_;
  cv::Mat last_img_;
  cv::Mat last_event_timestamp_;
  int image_height_;
  int image_width_;
};
