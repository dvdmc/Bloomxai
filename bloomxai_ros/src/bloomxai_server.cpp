#include "bloomxai_ros/bloomxai_server.hpp"

#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "bloomxai_ros/semantic_utils.hpp"

namespace {
template <typename T>
bool update_param(const std::vector<rclcpp::Parameter>& p, const std::string& name, T& value) {
  auto it = std::find_if(p.cbegin(), p.cend(), [&name](const rclcpp::Parameter& parameter) {
    return parameter.get_name() == name;
  });
  if (it != p.cend()) {
    value = it->template get_value<T>();
    return true;
  }
  return false;
}
}  // namespace

namespace bloomxai_server {
BloomxaiServer::BloomxaiServer(const rclcpp::NodeOptions& node_options)
    : Node("bloomxai_server_node", node_options) {
  using std::placeholders::_1;
  using std::placeholders::_2;

  {
    world_frame_id_ = declare_parameter("frame_id", "map");
    base_frame_id_ = declare_parameter("base_frame_id", "base_footprint"); // NOTE: Not used since we get it from PCD message
  }

  {
    rcl_interfaces::msg::ParameterDescriptor occupancy_min_z_desc;
    occupancy_min_z_desc.description =
        "Minimum height of occupied cells to consider in the final map";
    rcl_interfaces::msg::FloatingPointRange occupancy_min_z_range;
    occupancy_min_z_range.from_value = -100.0;
    occupancy_min_z_range.to_value = 100.0;
    occupancy_min_z_desc.floating_point_range.push_back(occupancy_min_z_range);
    occupancy_min_z_ = declare_parameter("occupancy_min_z", -4.0, occupancy_min_z_desc);
  }
  {
    rcl_interfaces::msg::ParameterDescriptor occupancy_max_z_desc;
    occupancy_max_z_desc.description =
        "Maximum height of occupied cells to consider in the final map";
    rcl_interfaces::msg::FloatingPointRange occupancy_max_z_range;
    occupancy_max_z_range.from_value = -100.0;
    occupancy_max_z_range.to_value = 100.0;
    occupancy_max_z_desc.floating_point_range.push_back(occupancy_max_z_range);
    occupancy_max_z_ = declare_parameter("occupancy_max_z", 15.0, occupancy_max_z_desc);
  }

  {
    rcl_interfaces::msg::ParameterDescriptor max_range_desc;
    max_range_desc.description = "Sensor maximum range";
    rcl_interfaces::msg::FloatingPointRange max_range_range;
    max_range_range.from_value = -1.0;
    max_range_range.to_value = 100.0;
    max_range_desc.floating_point_range.push_back(max_range_range);
    max_range_ = declare_parameter("sensor_model.max_range", -1.0, max_range_desc);
  }

  {
    rcl_interfaces::msg::ParameterDescriptor sem_dim_desc;
    sem_dim_desc.description = "Semantic dimension";
    rcl_interfaces::msg::IntegerRange sem_dim_range;
    sem_dim_range.from_value = 1;
    sem_dim_range.to_value = 1024;
    sem_dim_desc.integer_range.push_back(sem_dim_range);
    sem_dim_ = declare_parameter("sem_dim", 100, sem_dim_desc);
    initial_sem_val_ = 1.0f / sem_dim_;
    label_to_rgb_ = getLabelMap(sem_dim_);
  }

  res_ = declare_parameter("resolution", 0.1);

  rcl_interfaces::msg::ParameterDescriptor prob_hit_desc;
  prob_hit_desc.description =
      "Probabilities for hits in the sensor model when dynamically building a map";
  rcl_interfaces::msg::FloatingPointRange prob_hit_range;
  prob_hit_range.from_value = 0.5;
  prob_hit_range.to_value = 1.0;
  prob_hit_desc.floating_point_range.push_back(prob_hit_range);
  const double prob_hit = declare_parameter("sensor_model.hit", 0.7, prob_hit_desc);

  rcl_interfaces::msg::ParameterDescriptor prob_miss_desc;
  prob_miss_desc.description =
      "Probabilities for misses in the sensor model when dynamically building a map";
  rcl_interfaces::msg::FloatingPointRange prob_miss_range;
  prob_miss_range.from_value = 0.0;
  prob_miss_range.to_value = 0.5;
  prob_miss_desc.floating_point_range.push_back(prob_miss_range);
  const double prob_miss = declare_parameter("sensor_model.miss", 0.4, prob_miss_desc);

  rcl_interfaces::msg::ParameterDescriptor prob_min_desc;
  prob_min_desc.description = "Minimum probability for clamping when dynamically building a map";
  rcl_interfaces::msg::FloatingPointRange prob_min_range;
  prob_min_range.from_value = 0.0;
  prob_min_range.to_value = 1.0;
  prob_min_desc.floating_point_range.push_back(prob_min_range);
  const double thres_min = declare_parameter("sensor_model.min", 0.12, prob_min_desc);

  rcl_interfaces::msg::ParameterDescriptor prob_max_desc;
  prob_max_desc.description = "Maximum probability for clamping when dynamically building a map";
  rcl_interfaces::msg::FloatingPointRange prob_max_range;
  prob_max_range.from_value = 0.0;
  prob_max_range.to_value = 1.0;
  prob_max_desc.floating_point_range.push_back(prob_max_range);
  const double thres_max = declare_parameter("sensor_model.max", 0.97, prob_max_desc);

  // initialize bloomxai object & params
  RCLCPP_INFO(get_logger(), "Voxel resolution %f", res_);

  // Dump config
  RCLCPP_INFO(get_logger(), "occupancy_min_z %f", occupancy_min_z_);
  RCLCPP_INFO(get_logger(), "occupancy_max_z %f", occupancy_max_z_);
  RCLCPP_INFO(get_logger(), "max_range %f", max_range_);
  RCLCPP_INFO(get_logger(), "sem_dim %d", sem_dim_);
  RCLCPP_INFO(get_logger(), "initial_sem_val %f", initial_sem_val_);
  RCLCPP_INFO(get_logger(), "prob_hit %f", prob_hit);
  RCLCPP_INFO(get_logger(), "prob_miss %f", prob_miss);
  RCLCPP_INFO(get_logger(), "thres_min %f", thres_min);
  RCLCPP_INFO(get_logger(), "thres_max %f", thres_max);


  bloomxai_ = std::make_unique<BloomxaiT>(res_, sem_dim_);

  BloomxaiT::Options options(sem_dim_, initial_sem_val_);
  options.prob_miss_log = bloomxai_->logods(prob_miss);
  options.prob_hit_log = bloomxai_->logods(prob_hit);
  options.clamp_min_log = bloomxai_->logods(thres_min);
  options.clamp_max_log = bloomxai_->logods(thres_max);

  bloomxai_->setOptions(options);

  latched_topics_ = declare_parameter("latch", false);
  if (latched_topics_) {
    RCLCPP_INFO(
        get_logger(),
        "Publishing latched (single publish will take longer, "
        "all topics are prepared)");
  } else {
    RCLCPP_INFO(
        get_logger(),
        "Publishing non-latched (topics are only prepared as needed, "
        "will only be re-published on map change");
  }

  auto qos = latched_topics_ ? rclcpp::QoS{1}.transient_local() : rclcpp::QoS{1};
  point_cloud_pub_ = create_publisher<PointCloud2>("bloomxai_point_cloud_centers", qos);
  marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("bloomxai/marker", 1);

  tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
  auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
      this->get_node_base_interface(), this->get_node_timers_interface());
  tf2_buffer_->setCreateTimerInterface(timer_interface);
  tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);

  using std::chrono_literals::operator""s;
  point_cloud_sub_.subscribe(this, "cloud_in", rmw_qos_profile_default);
  tf_point_cloud_sub_ = std::make_shared<tf2_ros::MessageFilter<PointCloud2>>(
      point_cloud_sub_, *tf2_buffer_, world_frame_id_, 5, this->get_node_logging_interface(),
      this->get_node_clock_interface(), 5s);

  tf_point_cloud_sub_->registerCallback(&BloomxaiServer::insertCloudCallback, this);

  save_srv_ = create_service<bloomxai_ros::srv::SaveMap>(
  "~/save_map",
  std::bind(&BloomxaiServer::saveMapCallback, this, std::placeholders::_1, std::placeholders::_2));

  load_srv_ = create_service<bloomxai_ros::srv::LoadMap>(
  "~/load_map",
  std::bind(&BloomxaiServer::loadMapCallback, this, std::placeholders::_1, std::placeholders::_2));

  reset_srv_ =
      create_service<ResetSrv>("~/reset", std::bind(&BloomxaiServer::resetSrv, this, _1, _2));

  // set parameter callback
  set_param_res_ =
      this->add_on_set_parameters_callback(std::bind(&BloomxaiServer::onParameter, this, _1));
}

bool BloomxaiServer::saveMapCallback(
    const std::shared_ptr<bloomxai_ros::srv::SaveMap::Request> request,
    const std::shared_ptr<bloomxai_ros::srv::SaveMap::Response> response) {
  try {
    bloomxai_->serializeToFile(request->filename);
    RCLCPP_INFO(get_logger(), "Map saved to %s", request->filename.c_str());
    response->success = true;
    response->message = "Map saved successfully.";
    return true;
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_logger(), "Failed to save map: %s", e.what());
    response->success = false;
    response->message = e.what();
    return false;
  }
}

bool BloomxaiServer::loadMapCallback(
    const std::shared_ptr<bloomxai_ros::srv::LoadMap::Request> request,
    const std::shared_ptr<bloomxai_ros::srv::LoadMap::Response> response) {
  try {
    bloomxai_->deserializeFromFile(request->filename);
    RCLCPP_INFO(get_logger(), "Map loaded from %s", request->filename.c_str());
    response->success = true;
    response->message = "Map loaded successfully.";
    publishAll(now());
    return true;
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_logger(), "Failed to load map: %s", e.what());
    response->success = false;
    response->message = e.what();
    return false;
  }
}

void BloomxaiServer::insertCloudCallback(const PointCloud2::ConstSharedPtr cloud) {
  const auto start_time = rclcpp::Clock{}.now();
  
  RCLCPP_INFO(get_logger(), "Received cloud with %d points", cloud->width * cloud->height);

  PCLPointCloud pc;  // input cloud for filtering and ground-detection
  std::vector<Bloomxai::SemanticMap::VSemanticProb> semantics;

  sensor_msgs::PointCloud2ConstIterator<float> iter_x(*cloud, "x");
  sensor_msgs::PointCloud2ConstIterator<float> iter_y(*cloud, "y");
  sensor_msgs::PointCloud2ConstIterator<float> iter_z(*cloud, "z");
  sensor_msgs::PointCloud2ConstIterator<uint32_t> iter_rgba(*cloud, "rgb");
  sensor_msgs::PointCloud2ConstIterator<float> iter_gt_semantics(*cloud, "gt_semantics");
  sensor_msgs::PointCloud2ConstIterator<float> iter_semantics(*cloud, "semantics");

  float prob_max = 0.0;
  float prob_min = 1.0;
  for (; iter_x != iter_x.end();
       ++iter_x, ++iter_y, ++iter_z, ++iter_rgba, ++iter_gt_semantics, ++iter_semantics) {
    pcl::PointXYZ p;
    p.x = *iter_x;
    p.y = *iter_y;
    p.z = *iter_z;
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)
        || (p.x == 0.0 && p.y == 0.0 && p.z == 0.0)) {
      continue;
    }

    pc.points.push_back(p);
    Bloomxai::SemanticMap::VSemanticProb semantic(sem_dim_);
    for (int i = 0; i < sem_dim_; ++i) {
      semantic[i] = iter_semantics[i];
      if(semantic[i] > prob_max) {
        prob_max = semantic[i];
      }
      if(semantic[i] < prob_min) {
        prob_min = semantic[i];
      }
    };
    semantics.push_back(semantic);
  }

  // RCLCPP_INFO(get_logger(), "Check semantics: ");
  // for(int i = 0; i < semantics.size(); i++)
  // {

  //   for (int j = 0; j < sem_dim_; ++j)
  //   {
  
  //     std::cout << semantics[i][j] << " ";
  //   }
  //   std::cout << std::endl;
  // }


  RCLCPP_INFO(get_logger(), "Semantics sanity check: Max prob: %f, Min prob: %f", prob_max, prob_min);
  // Sensor In Global Frames Coordinates
  geometry_msgs::msg::TransformStamped sensor_to_world_transform_stamped;
  try {
    sensor_to_world_transform_stamped = tf2_buffer_->lookupTransform(
        world_frame_id_, cloud->header.frame_id, cloud->header.stamp,
        rclcpp::Duration::from_seconds(1.0));
  } catch (const tf2::TransformException& ex) {
    RCLCPP_WARN(this->get_logger(), "%s", ex.what());
    return;
  }

  Eigen::Matrix4f sensor_to_world =
      tf2::transformToEigen(sensor_to_world_transform_stamped.transform).matrix().cast<float>();

  // Transforming Points to Global Reference Frame
  pcl::transformPointCloud(pc, pc, sensor_to_world);

  // Getting the Translation from the sensor to the Global Reference Frame
  const auto& t = sensor_to_world_transform_stamped.transform.translation;

  RCLCPP_INFO(get_logger(), "Inserting %ld points", pc.points.size());

  const pcl::PointXYZ sensor_to_world_vec3(t.x, t.y, t.z);
  if (max_range_ >= 0) {
    bloomxai_->insertPointCloud(pc.points, semantics, sensor_to_world_vec3, max_range_);
  } else {
    bloomxai_->insertPointCloud(
        pc.points, semantics, sensor_to_world_vec3, std::numeric_limits<double>::infinity());
  }
  double total_elapsed = (rclcpp::Clock{}.now() - start_time).seconds();
  RCLCPP_INFO(get_logger(), "Pointcloud insertion in Bonxai done, %f sec)", total_elapsed);

  publishAll(cloud->header.stamp);
}

rcl_interfaces::msg::SetParametersResult BloomxaiServer::onParameter(
    const std::vector<rclcpp::Parameter>& parameters) {
  update_param(parameters, "occupancy_min_z", occupancy_min_z_);
  update_param(parameters, "occupancy_max_z", occupancy_max_z_);

  double sensor_model_min{get_parameter("sensor_model.min").as_double()};
  update_param(parameters, "sensor_model.min", sensor_model_min);
  double sensor_model_max{get_parameter("sensor_model.max").as_double()};
  update_param(parameters, "sensor_model.max", sensor_model_max);
  double sensor_model_hit{get_parameter("sensor_model.hit").as_double()};
  update_param(parameters, "sensor_model.hit", sensor_model_hit);
  double sensor_model_miss{get_parameter("sensor_model.miss").as_double()};
  update_param(parameters, "sensor_model.miss", sensor_model_miss);

  BloomxaiT::Options options(sem_dim_, initial_sem_val_);
  options.prob_miss_log = bloomxai_->logods(sensor_model_miss);
  options.prob_hit_log = bloomxai_->logods(sensor_model_hit);
  options.clamp_min_log = bloomxai_->logods(sensor_model_min);
  options.clamp_max_log = bloomxai_->logods(sensor_model_max);

  bloomxai_->setOptions(options);

  publishAll(now());

  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";
  return result;
}

void BloomxaiServer::publishAll(const rclcpp::Time& rostime) {
  const auto start_time = rclcpp::Clock{}.now();
  thread_local std::vector<Eigen::Vector3d> bloomxai_result;
  bloomxai_result.clear();
  thread_local std::vector<int> labels;
  labels.clear();
  bloomxai_->getOccupiedVoxelsAndClass(bloomxai_result, labels);

  if (bloomxai_result.size() <= 1) {
    RCLCPP_WARN(get_logger(), "Nothing to publish, bloomxai is empty");
    return;
  }

  bool publish_point_cloud =
      (latched_topics_ || point_cloud_pub_->get_subscription_count() +
                                  point_cloud_pub_->get_intra_process_subscription_count() >
                              0);

  // Publish Point Cloud
  if (publish_point_cloud) {
    thread_local pcl::PointCloud<PCLPointRGB> pcl_cloud;
    pcl_cloud.clear();

    for (size_t i = 0; i < bloomxai_result.size(); i++) {
      const auto& voxel = bloomxai_result[i];
      if (voxel.z() >= occupancy_min_z_ && voxel.z() <= occupancy_max_z_) {
        std::vector<uint8_t> color = label_to_rgb_[labels[i]];
        pcl_cloud.push_back(
            PCLPointRGB(voxel.x(), voxel.y(), voxel.z(), color[0], color[1], color[2]));
      }
    }

    PointCloud2 cloud;
    pcl::toROSMsg(pcl_cloud, cloud);
    cloud.header.frame_id = world_frame_id_;
    cloud.header.stamp = rostime;
    point_cloud_pub_->publish(cloud);

    RCLCPP_INFO(get_logger(), "Published occupancy grid with %ld voxels", pcl_cloud.points.size());
  }

  // Publish Cube Marker
  thread_local visualization_msgs::msg::Marker marker;
  marker.header.frame_id = world_frame_id_;
  marker.header.stamp = rostime;
  marker.ns = "bloomxai_voxels";
  marker.id = 0;
  marker.type = visualization_msgs::msg::Marker::CUBE_LIST;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = res_;  // size of voxel
  marker.scale.y = res_;
  marker.scale.z = res_;
  marker.points.clear();
  marker.colors.clear();

  for (size_t i = 0; i < bloomxai_result.size(); ++i) {
    const auto& voxel = bloomxai_result[i];
    if (voxel.z() >= occupancy_min_z_ && voxel.z() <= occupancy_max_z_) {
      geometry_msgs::msg::Point p;
      p.x = voxel.x();
      p.y = voxel.y();
      p.z = voxel.z();
      marker.points.push_back(p);

      std_msgs::msg::ColorRGBA color;
      auto rgb = label_to_rgb_[labels[i]];
      color.r = rgb[0] / 255.0f;
      color.g = rgb[1] / 255.0f;
      color.b = rgb[2] / 255.0f;
      color.a = 1.0f;
      marker.colors.push_back(color);
    }
  }

  if (marker_pub_->get_subscription_count() + marker_pub_->get_intra_process_subscription_count() > 0) {
    marker_pub_->publish(marker);
    RCLCPP_INFO(get_logger(), "Published marker with %ld cubes", marker.points.size());
  }
}

bool BloomxaiServer::resetSrv(
    const std::shared_ptr<ResetSrv::Request>, const std::shared_ptr<ResetSrv::Response>) {
  const auto rostime = now();
  bloomxai_ = std::make_unique<BloomxaiT>(res_, sem_dim_);

  RCLCPP_INFO(get_logger(), "Cleared Bonxai");
  publishAll(rostime);

  return true;
}

}  // namespace bloomxai_server

#include <rclcpp_components/register_node_macro.hpp>

RCLCPP_COMPONENTS_REGISTER_NODE(bloomxai_server::BloomxaiServer)