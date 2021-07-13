/*
 * test_patch_score.cpp
 *
 *  Created on: Dec 4, 2012
 *      Author: cforster
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <vikit/homography.h>
#include <vikit/math_utils.h>

namespace {

using namespace Eigen;

void loadData(
    vk::Bearings& f_ref,
    vk::Bearings& f_cur,
    double& focal_length)
{
  f_ref.resize(Eigen::NoChange,70);
  f_cur.resize(Eigen::NoChange,70);
  size_t i=0;
  f_ref.col(i++) = Vector3d(-0.275133, 0.139682, 0.951205);
  f_ref.col(i++) = Vector3d(-0.453144, 0.25876, 0.853056);
  f_ref.col(i++) = Vector3d(-0.683693, 0.252033, 0.684867);
  f_ref.col(i++) = Vector3d(0.0311563, 0.34865, 0.936735);
  f_ref.col(i++) = Vector3d(0.42499, 0.0768726, 0.901928);
  f_ref.col(i++) = Vector3d(-0.57448, 0.311981, 0.75673);
  f_ref.col(i++) = Vector3d(0.449507, 0.272281, 0.850768);
  f_ref.col(i++) = Vector3d(0.624694, 0.183813, 0.758927);
  f_ref.col(i++) = Vector3d(-0.43046, -0.178188, 0.884846);
  f_ref.col(i++) = Vector3d(-0.0532966, 0.346281, 0.936616);
  f_ref.col(i++) = Vector3d(0.588485, 0.229454, 0.775265);
  f_ref.col(i++) = Vector3d(-0.249363, 0.364956, 0.897009);
  f_ref.col(i++) = Vector3d(0.73978, 0.38238, 0.553635);
  f_ref.col(i++) = Vector3d(-0.512336, 0.363462, 0.778079);
  f_ref.col(i++) = Vector3d(0.169329, 0.419967, 0.891602);
  f_ref.col(i++) = Vector3d(0.770892, 0.274347, 0.574855);
  f_ref.col(i++) = Vector3d(0.108062, 0.37045, 0.922545);
  f_ref.col(i++) = Vector3d(0.380411, 0.284284, 0.88004);
  f_ref.col(i++) = Vector3d(-0.144588, 0.119588, 0.982239);
  f_ref.col(i++) = Vector3d(0.712077, 0.195253, 0.674406);
  f_ref.col(i++) = Vector3d(0.749152, 0.25336, 0.61203);
  f_ref.col(i++) = Vector3d(-0.38118, 0.337495, 0.860697);
  f_ref.col(i++) = Vector3d(-0.67239, 0.322924, 0.666041);
  f_ref.col(i++) = Vector3d(0.284355, 0.462661, 0.839695);
  f_ref.col(i++) = Vector3d(-0.322239, 0.46477, 0.824713);
  f_ref.col(i++) = Vector3d(0.0268799, 0.174692, 0.984256);
  f_ref.col(i++) = Vector3d(0.116106, 0.291004, 0.94965);
  f_ref.col(i++) = Vector3d(0.426098, 0.219401, 0.87767);
  f_ref.col(i++) = Vector3d(0.384375, 0.151817, 0.910608);
  f_ref.col(i++) = Vector3d(-0.641209, 0.342716, 0.686583);
  f_ref.col(i++) = Vector3d(-0.482429, 0.436318, 0.759532);
  f_ref.col(i++) = Vector3d(-0.127924, 0.35448, 0.926272);
  f_ref.col(i++) = Vector3d(0.45357, 0.159322, 0.876864);
  f_ref.col(i++) = Vector3d(-0.404163, 0.38604, 0.829232);
  f_ref.col(i++) = Vector3d(-0.452152, 0.355138, 0.81819);
  f_ref.col(i++) = Vector3d(0.368518, 0.355639, 0.858904);
  f_ref.col(i++) = Vector3d(-0.0863902, 0.31422, 0.945411);
  f_ref.col(i++) = Vector3d(0.534435, 0.392647, 0.74847);
  f_ref.col(i++) = Vector3d(0.236089, 0.452183, 0.860112);
  f_ref.col(i++) = Vector3d(0.131858, 0.45536, 0.880489);
  f_ref.col(i++) = Vector3d(-0.0431284, 0.467322, 0.883034);
  f_ref.col(i++) = Vector3d(-0.0620748, 0.415247, 0.907588);
  f_ref.col(i++) = Vector3d(0.361378, 0.407799, 0.838514);
  f_ref.col(i++) = Vector3d(0.547415, 0.234039, 0.803469);
  f_ref.col(i++) = Vector3d(0.656454, 0.258657, 0.708636);
  f_ref.col(i++) = Vector3d(0.706324, 0.272406, 0.653377);
  f_ref.col(i++) = Vector3d(0.160498, 0.407091, 0.899176);
  f_ref.col(i++) = Vector3d(-0.595751, -0.118818, 0.794332);
  f_ref.col(i++) = Vector3d(0.654654, 0.23348, 0.718968);
  f_ref.col(i++) = Vector3d(-0.464858, 0.350004, 0.813268);
  f_ref.col(i++) = Vector3d(-0.235892, 0.137801, 0.961959);
  f_ref.col(i++) = Vector3d(-0.247373, 0.147095, 0.95769);
  f_ref.col(i++) = Vector3d(0.620551, 0.192752, 0.760107);
  f_ref.col(i++) = Vector3d(0.546609, 0.385321, 0.743469);
  f_ref.col(i++) = Vector3d(0.40142, 0.174489, 0.899119);
  f_ref.col(i++) = Vector3d(-0.617385, 0.287266, 0.732335);
  f_ref.col(i++) = Vector3d(0.728777, -0.462694, 0.504775);
  f_ref.col(i++) = Vector3d(0.69319, 0.171178, 0.700132);
  f_ref.col(i++) = Vector3d(-0.089887, 0.41494, 0.905398);
  f_ref.col(i++) = Vector3d(0.450678, 0.245466, 0.858275);
  f_ref.col(i++) = Vector3d(-0.0917591, 0.235608, 0.967507);
  f_ref.col(i++) = Vector3d(-0.277634, 0.251282, 0.927241);
  f_ref.col(i++) = Vector3d(-0.60967, 0.29214, 0.736856);
  f_ref.col(i++) = Vector3d(0.651581, 0.156287, 0.742305);
  f_ref.col(i++) = Vector3d(0.731917, 0.303376, 0.610131);
  f_ref.col(i++) = Vector3d(0.707427, 0.326456, 0.626876);
  f_ref.col(i++) = Vector3d(0.455411, 0.320352, 0.830648);
  f_ref.col(i++) = Vector3d(-0.00165953, 0.461063, 0.887366);
  f_ref.col(i++) = Vector3d(-0.51092, 0.260245, 0.819288);
  f_ref.col(i++) = Vector3d(-0.562318, 0.321435, 0.761891);
  i = 0;
  f_cur.col(i++) = Vector3d(-0.314108, 0.0787836, 0.946113);
  f_cur.col(i++) = Vector3d(-0.49336, 0.201209, 0.846233);
  f_cur.col(i++) = Vector3d(-0.714299, 0.203437, 0.669619);
  f_cur.col(i++) = Vector3d(-0.0124058, 0.287466, 0.95771);
  f_cur.col(i++) = Vector3d(0.391199, 0.0146728, 0.920189);
  f_cur.col(i++) = Vector3d(-0.613535, 0.257692, 0.746438);
  f_cur.col(i++) = Vector3d(0.418711, 0.216758, 0.881871);
  f_cur.col(i++) = Vector3d(0.601104, 0.13628, 0.787465);
  f_cur.col(i++) = Vector3d(-0.467537, -0.244271, 0.849553);
  f_cur.col(i++) = Vector3d(-0.0978916, 0.284652, 0.95362);
  f_cur.col(i++) = Vector3d(0.562768, 0.179693, 0.806847);
  f_cur.col(i++) = Vector3d(-0.295436, 0.304041, 0.905691);
  f_cur.col(i++) = Vector3d(0.726236, 0.347243, 0.593299);
  f_cur.col(i++) = Vector3d(-0.553105, 0.308215, 0.774002);
  f_cur.col(i++) = Vector3d(0.12876, 0.363061, 0.922826);
  f_cur.col(i++) = Vector3d(0.755376, 0.237074, 0.610903);
  f_cur.col(i++) = Vector3d(0.0658183, 0.310916, 0.948156);
  f_cur.col(i++) = Vector3d(0.347842, 0.228711, 0.909229);
  f_cur.col(i++) = Vector3d(-0.18349, 0.0577989, 0.981321);
  f_cur.col(i++) = Vector3d(0.692312, 0.152672, 0.705263);
  f_cur.col(i++) = Vector3d(0.732262, 0.214083, 0.646499);
  f_cur.col(i++) = Vector3d(-0.424814, 0.278522, 0.86137);
  f_cur.col(i++) = Vector3d(-0.705486, 0.273901, 0.653657);
  f_cur.col(i++) = Vector3d(0.247401, 0.409651, 0.878054);
  f_cur.col(i++) = Vector3d(-0.369072, 0.407299, 0.8354);
  f_cur.col(i++) = Vector3d(-0.0171458, 0.107884, 0.994016);
  f_cur.col(i++) = Vector3d(0.0746287, 0.229733, 0.970388);
  f_cur.col(i++) = Vector3d(0.39396, 0.161353, 0.904854);
  f_cur.col(i++) = Vector3d(0.350229, 0.0951773, 0.931816);
  f_cur.col(i++) = Vector3d(-0.675607, 0.290697, 0.677532);
  f_cur.col(i++) = Vector3d(-0.525243, 0.381283, 0.760751);
  f_cur.col(i++) = Vector3d(-0.173641, 0.292971, 0.940222);
  f_cur.col(i++) = Vector3d(0.421607, 0.10038, 0.901205);
  f_cur.col(i++) = Vector3d(-0.448002, 0.328861, 0.831351);
  f_cur.col(i++) = Vector3d(-0.494992, 0.298149, 0.816143);
  f_cur.col(i++) = Vector3d(0.335083, 0.300948, 0.892832);
  f_cur.col(i++) = Vector3d(-0.130437, 0.251905, 0.958921);
  f_cur.col(i++) = Vector3d(0.508842, 0.344173, 0.789065);
  f_cur.col(i++) = Vector3d(0.198171, 0.398371, 0.895561);
  f_cur.col(i++) = Vector3d(0.0902021, 0.399366, 0.912343);
  f_cur.col(i++) = Vector3d(-0.0796933, 0.408578, 0.909238);
  f_cur.col(i++) = Vector3d(-0.107955, 0.355663, 0.928359);
  f_cur.col(i++) = Vector3d(0.328433, 0.355416, 0.875106);
  f_cur.col(i++) = Vector3d(0.52189, 0.182516, 0.833258);
  f_cur.col(i++) = Vector3d(0.635367, 0.213799, 0.742023);
  f_cur.col(i++) = Vector3d(0.687613, 0.23057, 0.688495);
  f_cur.col(i++) = Vector3d(0.119683, 0.349575, 0.929233);
  f_cur.col(i++) = Vector3d(-0.629833, -0.179005, 0.755822);
  f_cur.col(i++) = Vector3d(0.633002, 0.187888, 0.751004);
  f_cur.col(i++) = Vector3d(-0.507273, 0.293347, 0.810322);
  f_cur.col(i++) = Vector3d(-0.27525, 0.0768303, 0.958298);
  f_cur.col(i++) = Vector3d(-0.286753, 0.0861832, 0.95412);
  f_cur.col(i++) = Vector3d(0.596872, 0.145137, 0.7891);
  f_cur.col(i++) = Vector3d(0.521313, 0.33688, 0.784057);
  f_cur.col(i++) = Vector3d(0.367602, 0.11442, 0.922918);
  f_cur.col(i++) = Vector3d(-0.652129, 0.235195, 0.720702);
  f_cur.col(i++) = Vector3d(0.711624, -0.494716, 0.498846);
  f_cur.col(i++) = Vector3d(0.671912, 0.127027, 0.729656);
  f_cur.col(i++) = Vector3d(-0.135909, 0.355126, 0.924886);
  f_cur.col(i++) = Vector3d(0.41936, 0.1895, 0.887821);
  f_cur.col(i++) = Vector3d(-0.134665, 0.173412, 0.975599);
  f_cur.col(i++) = Vector3d(-0.320945, 0.189392, 0.927968);
  f_cur.col(i++) = Vector3d(-0.644542, 0.239822, 0.725983);
  f_cur.col(i++) = Vector3d(0.628599, 0.109822, 0.769937);
  f_cur.col(i++) = Vector3d(0.715956, 0.263814, 0.646382);
  f_cur.col(i++) = Vector3d(0.690093, 0.286395, 0.664642);
  f_cur.col(i++) = Vector3d(0.425479, 0.267033, 0.864674);
  f_cur.col(i++) = Vector3d(-0.0358689, 0.402047, 0.914916);
  f_cur.col(i++) = Vector3d(-0.548987, 0.204038, 0.810544);
  f_cur.col(i++) = Vector3d(-0.601498, 0.266629, 0.753066);
  focal_length = 386.711;
}

TEST(Homography, testHomography)
{
  vk::Bearings f_ref, f_cur;
  double focal_length;
  double reproj_error_thresh = 2.0;
  loadData(f_ref, f_cur, focal_length);
  vk::Homography H = vk::estimateHomography(
        f_cur, f_ref, focal_length, reproj_error_thresh, 0.0);

  // compute inliers
  std::vector<int> inliers, outliers;
  vk::Bearings xyz_in_cur;
  vk::computeInliers(
        f_cur, f_ref, H.R_cur_ref, H.t_cur_ref, reproj_error_thresh,
        focal_length, xyz_in_cur, inliers, outliers);

  EXPECT_EQ(inliers.size(), 69);
}

} // namespace

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
