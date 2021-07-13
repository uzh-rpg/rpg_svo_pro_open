#include <iostream>
#include <Eigen/Dense>

void testEuclideanNorm()
{
  Eigen::Matrix<double, 3, 5> A;
  A << 1, 2, 3, 4, 5,
       2, 3, 4, 5, 6,
       7, 9, 10, 11, 12;

  std::cout << A.colwise().norm() << std::endl << std::endl;

  Eigen::Matrix<double, 3, 5> B;
  B = A.array().rowwise() / A.colwise().norm().array();
  std::cout << B << std::endl;

  for(int i=0; i<A.cols(); ++i)
  {
    Eigen::Vector3d b = A.col(i)/A.col(i).norm();
    std::cout << b.transpose() << std::endl;
    std::cout << b.norm() << std::endl;
  }

}

int main(int argc, char** argv)
{
  testEuclideanNorm();
  return 0;
}
