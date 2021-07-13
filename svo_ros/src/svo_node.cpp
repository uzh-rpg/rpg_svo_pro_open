#include "svo_ros/svo_node_base.h"

int main(int argc, char **argv)
{
  svo_ros::SvoNodeBase::initThirdParty(argc, argv);

  svo_ros::SvoNodeBase node;
  node.run();
}
