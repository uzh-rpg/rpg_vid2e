#include <esim.h>

#include <fstream>
#include <iostream>
#include <algorithm>

#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


EventSimulator::EventSimulator(float contrast_threshold_pos, 
                               float contrast_threshold_neg,
                               float refractory_period,
                               float log_eps,
                               bool use_log_img)
    : contrast_threshold_pos_(contrast_threshold_pos), contrast_threshold_neg_(contrast_threshold_neg),
    refractory_period_(refractory_period), log_eps_(log_eps), use_log_img_(use_log_img), is_initialized_(false)
{

}

Eigen::MatrixXd EventSimulator::generateFromVideo(std::string video_path, std::string timestamps_file_path)
{
    std::ifstream timestamps_file(timestamps_file_path);
    
    if(!timestamps_file.is_open()) 
        throw std::runtime_error("unable to open the file " + timestamps_file_path);

    cv::VideoCapture cap(video_path);

    if ( !cap.isOpened() ) 
        throw std::runtime_error("Cannot open the video file " + video_path);

    std::string time_str;
    double time;

    std::vector<Event> events_vec;
    
    cv::Mat img, log_img;

    while (cap.read(img))
    {
        img.convertTo(img, CV_32F, 1.0/255);
        cv::Mat log_img = img;
        if (use_log_img_)
            cv::log(img+log_eps_, log_img);
        
        std::getline(timestamps_file, time_str);
        time = std::stod(time_str);
        
        imageCallback(log_img, time, events_vec);
    }

    // reset state to generate new events
    is_initialized_ = false;

    return vec_to_eigen_matrix(events_vec);
}

Eigen::MatrixXd EventSimulator::generateFromStampedImageSequence(std::vector<std::string> image_paths, std::vector<double> timestamps)
{
    // check that timestamps are ascending
    if (image_paths.size() != timestamps.size())
        throw std::runtime_error("Number of image paths and number of timestamps should be equal. Got " + std::to_string(image_paths.size()) + " and " + std::to_string(timestamps.size()));

    cv::Mat img, log_img;
    double time;

    std::vector<Event> events_vec;

    for (int i=0; i<timestamps.size(); i++)
    {
        if ((i < timestamps.size()-1)  && timestamps[i+1]<timestamps[i]) 
            throw std::runtime_error("Timestamps must be sorted in ascending order.");

        img = cv::imread(image_paths[i], cv::IMREAD_GRAYSCALE);

        if(img.empty()) 
            throw std::runtime_error("unable to open the image " + image_paths[i]);
       
        img.convertTo(img, CV_32F, 1.0/255);
        cv::Mat log_img = img;
        if (use_log_img_)
            cv::log(img+log_eps_, log_img);

        time = timestamps[i];

        imageCallback(log_img, time, events_vec);
    }

    // reset state to generate new events
    is_initialized_ = false;

    return vec_to_eigen_matrix(events_vec);
}


Eigen::MatrixXd EventSimulator::generateFromFolder(std::string image_folder, std::string timestamps_file_path)
{
    std::vector<std::string> image_files;
    read_directory_from_path(image_folder, image_files);
    std::ifstream timestamps_file(timestamps_file_path);
    
    if(!timestamps_file.is_open()) 
        throw std::runtime_error("unable to open the file " + timestamps_file_path);

    std::string time_str;
    double time;

    std::vector<Event> events_vec;
    
    cv::Mat img, log_img;

    for (const std::string& file : image_files)
    {
        img = cv::imread(file, cv::IMREAD_GRAYSCALE);
        if(img.empty()) 
            throw std::runtime_error("unable to open the image " + file);

        img.convertTo(img, CV_32F, 1.0/255);
        cv::Mat log_img = img;
        if (use_log_img_)
            cv::log(img+log_eps_, log_img);

        std::getline(timestamps_file, time_str);
        time = std::stod(time_str);

        imageCallback(log_img, time, events_vec);
    }

    // reset state to generate new events
    is_initialized_ = false;

    return vec_to_eigen_matrix(events_vec);
}


void EventSimulator::init(const cv::Mat &img, double time)
{
  is_initialized_ = true;
  last_img_ = img;
  ref_values_ = img;

  last_event_timestamp_ = cv::Mat::zeros(img.size[0], img.size[1], CV_64F);

  current_time_ = time;
  image_width_ = img.size[1];
  image_height_ = img.size[0];
}

void EventSimulator::imageCallback(const cv::Mat& img, double time, std::vector<Event>& events)
{
    cv::Mat preprocessed_img = img;
  
    if(!is_initialized_)
    {
        init(preprocessed_img, time);
        return;
    }

    std::vector<Event> new_events;

    static constexpr double kTolerance = 1e-6;
    double delta_t = time - current_time_;

    for (int y = 0; y < image_height_; ++y)
    {
        for (int x = 0; x < image_width_; ++x)
        {
            float& itdt = preprocessed_img.at<float>(y, x);
            float& it = last_img_.at<float>(y, x);
            float& prev_cross = ref_values_.at<float>(y, x);

            if (std::fabs (it - itdt) > kTolerance)
            {
                float pol = (itdt >= it) ? +1.0 : -1.0;
                float C = (pol > 0) ? contrast_threshold_pos_ : contrast_threshold_neg_;
                
                float curr_cross = prev_cross;
                bool all_crossings = false;
      
                do
                {
                    curr_cross += pol * C;
      
                    if ((pol > 0 && curr_cross > it && curr_cross <= itdt)
                        || (pol < 0 && curr_cross < it && curr_cross >= itdt))
                    {
                        const double edt = (curr_cross - it) * delta_t / (itdt - it);
                        const double t = current_time_ + edt;
      
                        const double last_stamp_at_xy = last_event_timestamp_.at<double>(y,x);
                        
                        const double dt = t - last_stamp_at_xy;

                        if(last_stamp_at_xy == 0 || dt >= refractory_period_)
                        {
                            new_events.emplace_back(x, y,t,pol);
                            last_event_timestamp_.at<double>(y,x) = t;
                        }
                     
                        ref_values_.at<float>(y,x) = curr_cross;
                    }
                    else
                    {
                        all_crossings = true;
                    }
                } while (!all_crossings);
            } // end tolerance
        } // end for each pixel
    }

    current_time_ = time;
    last_img_ = preprocessed_img; // it is now the latest image

    // need to sort the new events before inserting
    std::sort(new_events.begin(), new_events.end());
    events.insert(events.end(), new_events.begin(), new_events.end());
}
