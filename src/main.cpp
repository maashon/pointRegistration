#include <numeric>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <chrono>
#include <omp.h>
#include <iterator>
#include <random>

#include "../nanoflann.hpp"

using namespace cv;
using namespace Eigen;
using namespace nanoflann;
using namespace std;




typedef KDTreeEigenMatrixAdaptor<MatrixXd> my_kd_tree_t;
vector<int> indices;
vector<double> distances;
vector<tuple<int, int, double>> pairWiseDistances;

typedef struct Transformations {
    Mat Rotation;
    Mat translation;
} Trafo;

typedef struct dimention {
    int row;
    int col;
}dimention;

dimention getDimention(string address) {
    dimention d;
    ifstream MyReadFile(address);
    //string t;
    size_t lines = 0;
    for (std::string t; getline(MyReadFile, t);) {
        
        lines = lines + 1;
    }
    MyReadFile.close();
    d.col = 3;
    d.row = lines;

    return d;
}



void readPointCloud(MatrixXd& M1, MatrixXd& M2, string address1, string address2) {
   

    //reading the first pointcload
    dimention d1 = getDimention(address1);
    M1.resize(d1.row,d1.col );
    string t;
    double x, y, z;
    ifstream PC1(address1);
    cout << "reading points from the first xyz file\n";
    size_t i = 0;
    while (getline(PC1, t)) {
        x = stod(t.substr(0, t.find(" ")));
        t.erase(0, t.find(" ") + 1);
        y = stod(t.substr(0, t.find(" ")));
        t.erase(0, t.find(" ") + 1);
        z = stod(t.substr(0, t.length()));
      
        M1(i, 0) = x;
        M1(i, 1) = y;
        M1(i, 2) = z;
        
        i++;
    }
    PC1.close();
    std::cout << "first Point cloud loaded\n";


    //reading the second point cload.
    dimention d2 = getDimention(address2);
    M2.resize(d2.row , d2.col);
    ifstream PC2(address2);
    cout << "reading points from the second xyz file\n";
    i = 0;
    while (getline(PC2, t)) {
        x = stod(t.substr(0, t.find(" ")));
        t.erase(0, t.find(" ") + 1);
        y = stod(t.substr(0, t.find(" ")));
        t.erase(0, t.find(" ") + 1);
        z = stod(t.substr(0, t.length()));
        M2(i, 0) = x;
        M2(i, 1) = y;
        M2(i, 2) = z;
        i++;
    }
    PC2.close();
    std::cout << "first Point cloud loaded\n";
    std::cout << "Point reading done!\n";
    
}




void findNearest(Mat mat1, const my_kd_tree_t& kd, const size_t rowNum, const size_t columnNum) {
  indices.clear();
  distances.clear();
  for (int i = 0; i < rowNum; ++i) {
    // Query point:
    vector<double> query_pt(columnNum);
    for (size_t d = 0; d < columnNum; d++) {
        query_pt[d] = mat1.at<double>(i, d);
    }
    // do a knn search
    size_t index;
    double dist;

    nanoflann::KNNResultSet<double> resultSet(1);
    resultSet.init(&index, &dist);

    kd.index->findNeighbors(resultSet, &query_pt[0],nanoflann::SearchParams(10));
    indices.emplace_back(int(index));
    distances.emplace_back(dist);
  }
}

Transformations closedFormSVD(const Mat& pointcloud_1, const Mat& pointcloud_2) {
    Point3d center1, center2;
    Mat R, T;
    R = Mat::zeros(3, 3, CV_64F);
    T = Mat::zeros(3, 1, CV_64F);

    Mat cov = Mat::zeros(3, 3, CV_64F);
    vector<Point3d> pointCloud1Center, pointCloud2Center;
    Transformations ret;

    int rows = pointcloud_1.rows, pcl2_rows = pointcloud_2.rows;
    // Calculate the center of the first point cload
    for (int i = 0; i < rows; i++) {
        center1.x += pointcloud_1.at<double>(i, 0);
        center1.y += pointcloud_1.at<double>(i, 1);
        center1.z += pointcloud_1.at<double>(i, 2);
    }
    center1 = center1 * (1.0 / rows);

    // Calculate the center of second point cload
    for (int i = 0; i < pcl2_rows; i++) {
        center2.x += pointcloud_2.at<double>(i, 0);
        center2.y += pointcloud_2.at<double>(i, 1);
        center2.z += pointcloud_2.at<double>(i, 2);
    }
    center2 = center2 * (1.0 / pcl2_rows);

    // Move first point cload to the center
    for (int i = 0; i < rows; i++) {
        Point3d pt;
        pt.x = pointcloud_1.at<double>(i, 0) - center1.x;
        pt.y = pointcloud_1.at<double>(i, 1) - center1.y;
        pt.z = pointcloud_1.at<double>(i, 2) - center1.z;
        pointCloud1Center.emplace_back(pt);
    }

    // Move second pointcload to the center
    for (int i = 0; i < pcl2_rows; i++) {
        Point3d pt;
        pt.x = pointcloud_2.at<double>(i, 0) - center2.x;
        pt.y = pointcloud_2.at<double>(i, 1) - center2.y;
        pt.z = pointcloud_2.at<double>(i, 2) - center2.z;
        pointCloud2Center.emplace_back(pt);
    }

    // calcualting the covariance matrix with the obtained matrices
    for (int i = 0; i < rows; i++) {
        cov.at<double>(0, 0) += pointCloud1Center[i].x * pointCloud2Center[i].x;
        cov.at<double>(0, 1) += pointCloud1Center[i].x * pointCloud2Center[i].y;
        cov.at<double>(0, 2) += pointCloud1Center[i].x * pointCloud2Center[i].z;
        cov.at<double>(1, 0) += pointCloud1Center[i].y * pointCloud2Center[i].x;
        cov.at<double>(1, 1) += pointCloud1Center[i].y * pointCloud2Center[i].y;
        cov.at<double>(1, 2) += pointCloud1Center[i].y * pointCloud2Center[i].z;
        cov.at<double>(2, 0) += pointCloud1Center[i].z * pointCloud2Center[i].x;
        cov.at<double>(2, 1) += pointCloud1Center[i].z * pointCloud2Center[i].y;
        cov.at<double>(2, 2) += pointCloud1Center[i].z * pointCloud2Center[i].z;
    }
    cov /= rows;
    Mat U, D, Vtranspose;
    
    SVD::compute(cov, U, D, Vtranspose, 0);

    // Calculate the rotation matrix
    R = Vtranspose.t() * D.t();
    if (determinant(R) < 0.) {
        Vtranspose.at<double>(2, 0) *= -1;
        Vtranspose.at<double>(2, 1) *= -1;
        Vtranspose.at<double>(2, 2) *= -1;
        R = Vtranspose.t() * D.t();
    }

    Mat X0 = Mat::zeros(3, 1, CV_64F); 
    X0.at<double>(0, 0) = center1.x;
    X0.at<double>(1, 0) = center1.y;
    X0.at<double>(2, 0) = center1.z;

    Mat Y0 = Mat::zeros(3, 1, CV_64F);
    Y0.at<double>(0, 0) = center2.x;
    Y0.at<double>(1, 0) = center2.y;
    Y0.at<double>(2, 0) = center2.z;

    // Calculating the translation matrix t=Y0 -R*X0
    T = Y0 - (R * X0);

    ret.Rotation = R;
    ret.translation = T;
    return ret;
}

bool sortPoints(const tuple<int, int, double>& a, const tuple<int, int, double>& b){
    return (get<2>(a) < get<2>(b));
}

Matrix3d createRotatoinMatrix(vector<double>& theta)
{
    Matrix3d R_x ;
    // Calculate rotation about x axis
    R_x <<
        1, 0, 0,
        0, cos(theta[0]), -sin(theta[0]),
        0, sin(theta[0]), cos(theta[0]);

    // Calculate rotation about y axis
    Matrix3d R_y ;
    R_y <<
        cos(theta[1]), 0, sin(theta[1]),
        0, 1, 0,
        -sin(theta[1]), 0, cos(theta[1]);

    // Calculate rotation about z axis
    Matrix3d R_z ;
    R_z <<
        cos(theta[2]), -sin(theta[2]), 0,
        sin(theta[2]), cos(theta[2]), 0,
        0, 0, 1;

    // Combined rotation matrix
    return R_z * R_y * R_x;
}


void saveRotated(MatrixXd& M, vector<double>& theta, string address) {
    ofstream file(address);
    Matrix3d R = createRotatoinMatrix(theta);


    if (file.is_open()) {
#pragma omp parallel for
    for (int i = 0; i < M.rows(); i++) {
        Vector3d pt;
        pt.x() = M(i, 0);
        pt.y() = M(i, 1);
        pt.z() = M(i, 2);

        Vector3d p = R * pt;
        file << p.x() << " " << p.y() << " " << p.z() << endl;

    }
    }

    file.close();

}
double calculateError(Mat mat1, Mat mat2, Mat rot, Mat transl) {
    double error = 0;
    for (int i = 0; i < mat2.rows ; i++) {
        Mat pont1 = Mat::zeros(3, 1, CV_64F);
        Mat pont2 = Mat::zeros(3, 1, CV_64F);
        pont1.at<double>(0, 0) = mat1.at<double>(i, 0);
        pont1.at<double>(1, 0) = mat1.at<double>(i, 1);
        pont1.at<double>(2, 0) = mat1.at<double>(i, 2);
        pont2.at<double>(0, 0) = mat2.at<double>(i, 0);
        pont2.at<double>(1, 0) = mat2.at<double>(i, 1);
        pont2.at<double>(2, 0) = mat2.at<double>(i, 2);
        Mat diff = (rot * pont1 + transl) - pont2;
        error += sqrt(diff.at<double>(0, 0) * diff.at<double>(0, 0) + diff.at<double>(1, 0) * diff.at<double>(1, 0) + diff.at<double>(2, 0) * diff.at<double>(2, 0));
    }
    error = error / mat2.rows;
    return error;
}

double calculateErrorTrimmed(int NPo, vector<tuple<int, int, double>> pairedIndicesAndDistances) {
    double MSE = 0.;
    int length = NPo;
    for (int i = 0; i < length; i++) {
        MSE += get<2>(pairedIndicesAndDistances[i]);
    }
    MSE /= length;
    return MSE;
}

void saveRegistratoinResultRGB(Mat& Mat1, Mat& Mat2, string address) {
    ofstream file(address);
    if (file.is_open())
    {
        for (int i = 0; i < Mat1.rows; ++i) {
            file << Mat1.at<double>(i, 0) << " " << Mat1.at<double>(i, 1) << " " << Mat1.at<double>(i, 2) <<" "<<255 << " " <<0 << " " <<0<< endl;
        }
        for (int i = 0; i < Mat2.rows; ++i) {
            file << Mat2.at<double>(i, 0) << " " << Mat2.at<double>(i, 1) << " " << Mat2.at<double>(i, 2) << " " << 0 << " " << 0 << " " << 255 << endl;
        }
    }
    file.close();


}

void saveRegistratoinResult(Mat& Mat1, Mat& Mat2, string address) {
    ofstream file(address);
    if (file.is_open())
    {
        for (int i = 0; i < Mat1.rows; ++i) {
            file << Mat1.at<double>(i, 0) << " " << Mat1.at<double>(i, 1) << " " << Mat1.at<double>(i, 2) << endl;
        }
        for (int i = 0; i < Mat2.rows; ++i) {
            file << Mat2.at<double>(i, 0) << " " << Mat2.at<double>(i, 1) << " " << Mat2.at<double>(i, 2) << endl;
        }
    }
    file.close();


}

void addGaussianNoise(MatrixXd& M,string address,const double mean, const double std) {
    ofstream file(address);
    std::default_random_engine generator;
    std::normal_distribution<double> dist(mean, std);
    double x, y, z;
    for (int i = 0; i < M.rows(); ++i) {
        x = M(i, 0) + dist(generator);
        y = M(i, 1) + dist(generator);
        z = M(i, 2) + dist(generator);
        file << x << " " << y<< " " << z << endl;
    }
    file.close();
}

void icp(MatrixXd M1, MatrixXd M2, int max_iteration_num) {
    auto time_start = std::chrono::high_resolution_clock::now();
    Transformations T;
    
    
    double lastError = 0.0;
    double newError = 0.0;
    double tol = 0.00001;
    
    //building KD tree
    my_kd_tree_t kd(3, std::cref(M2), 10);
    kd.index->buildIndex();
    Mat rotation_matrix = Mat::eye(3, 3, CV_64F);
    Mat translation_matrix = Mat::eye(3, 1, CV_64F);
    std::cout<<"ICP Mean Errors:\n"<< endl;

    Mat pcSource, pcTarget;
    eigen2cv(M1, pcSource);
    eigen2cv(M2, pcTarget);
    for (int iteratoins = 0; iteratoins < max_iteration_num; iteratoins++) {
        findNearest(pcSource, kd, pcSource.rows, 3);
        Mat newTarget = Mat::zeros(pcSource.rows, 3, CV_64F);
        for (int i = 0; i < pcSource.rows; i++) {
            newTarget.at<double>(i, 0) = pcTarget.at<double>(indices[i], 0);
            newTarget.at<double>(i, 1) = pcTarget.at<double>(indices[i], 1);
            newTarget.at<double>(i, 2) = pcTarget.at<double>(indices[i], 2);
        }
        
        T = closedFormSVD(pcSource, newTarget);
       
        // Compute motion that minimises mean square error(MSE) between paired points.
        rotation_matrix *= T.Rotation;
        translation_matrix += T.translation;
        
        for (int i = 0; i < pcSource.rows; i++) {     // Apply motion to P and update MSE.
            Mat pont = Mat::zeros(3, 1, CV_64F);
            pont.at<double>(0, 0) = pcSource.at<double>(i, 0);
            pont.at<double>(1, 0) = pcSource.at<double>(i, 1);
            pont.at<double>(2, 0) = pcSource.at<double>(i, 2);
            pont = T.Rotation * pont;
            pont += T.translation;
            pcSource.at<double>(i, 0) = pont.at<double>(0, 0);
            pcSource.at<double>(i, 1) = pont.at<double>(1, 0);
            pcSource.at<double>(i, 2) = pont.at<double>(2, 0);
        }

        newError = calculateError(pcSource, newTarget, T.Rotation, T.translation); // Updating MSE
        std::cout << newError << endl;
        if (abs(lastError - newError) < tol) {
            break;
        }
        lastError = newError;
    }
    auto time_finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = (time_finish - time_start);
    std::cout << "ICP Execution Time = " << duration.count() << " s" << endl;
    std::cout << "Rotation matrix: " << endl << rotation_matrix << endl;
    std::cout << "Translation matrix: " << endl << translation_matrix << endl;

    saveRegistratoinResultRGB(pcSource, pcTarget, "../ICP_RGB.txt");
    saveRegistratoinResultRGB(pcSource, pcTarget, "../ICP.xyz");
    
    
    
}

void tricp(MatrixXd M1, MatrixXd M2, int max_iteration_num) {
    auto time_start = std::chrono::high_resolution_clock::now();
    Transformations T;
    Mat cvSource, cvTarget;
    eigen2cv(M1, cvSource);
    eigen2cv(M2, cvTarget);
    double prev_error = 0.0;
    double mean_error = 0.0;
    double tolerance = 0.0001;
   
    my_kd_tree_t mat_index(3, std::cref(M2), 10 /* max leaf */);
    Mat rotation_matrix = Mat::eye(3, 3, CV_64F);
    Mat translation_matrix = Mat::eye(3, 1, CV_64F);
    mat_index.index->buildIndex();
    std::cout << "TR-ICp MSE:\n";
    for (int iters_ = 0; iters_ < max_iteration_num; iters_++) {
        pairWiseDistances.clear();
        findNearest(cvSource, mat_index, cvSource.rows, 3);

        for (int i = 0; i < cvSource.rows; i++) {
            pairWiseDistances.push_back(make_tuple(i, indices[i], distances[i]));
        }

        sort(pairWiseDistances.begin(), pairWiseDistances.end(), sortPoints);
        sort(distances.begin(), distances.end());
        int NPo = 0.6 * double(pairWiseDistances.size());
        Mat cvNewSource = Mat::zeros(NPo, 3, CV_64F);
        Mat cvNewTarget = Mat::zeros(NPo, 3, CV_64F);
        for (int i = 0; i < NPo; i++) {
            cvNewSource.at<double>(i, 0) = cvSource.at<double>(get<0>(pairWiseDistances[i]), 0);
            cvNewSource.at<double>(i, 1) = cvSource.at<double>(get<0>(pairWiseDistances[i]), 1);
            cvNewSource.at<double>(i, 2) = cvSource.at<double>(get<0>(pairWiseDistances[i]), 2);
            cvNewTarget.at<double>(i, 0) = cvTarget.at<double>(get<1>(pairWiseDistances[i]), 0);
            cvNewTarget.at<double>(i, 1) = cvTarget.at<double>(get<1>(pairWiseDistances[i]), 1);
            cvNewTarget.at<double>(i, 2) = cvTarget.at<double>(get<1>(pairWiseDistances[i]), 2);
        }

        T = closedFormSVD(cvNewSource, cvNewTarget); // For !!!! Npo selected pairs !!!!, compute optimal motion(R, t) that minimises STS
        rotation_matrix *= T.Rotation;
        translation_matrix += T.translation;
        for (int i = 0; i < cvSource.rows; i++) {     // Apply motion to P and update MSE.
            Mat pont = Mat::zeros(3, 1, CV_64F);
            pont.at<double>(0, 0) = cvSource.at<double>(i, 0);
            pont.at<double>(1, 0) = cvSource.at<double>(i, 1);
            pont.at<double>(2, 0) = cvSource.at<double>(i, 2);
            pont = T.Rotation * pont;
            pont += T.translation;
            cvSource.at<double>(i, 0) = pont.at<double>(0, 0);
            cvSource.at<double>(i, 1) = pont.at<double>(1, 0);
            cvSource.at<double>(i, 2) = pont.at<double>(2, 0);
        }
        // mean_error = calculateError(cvSource, cvNewTarget, T.rot, T.transl); // Updating MSE
        mean_error = calculateErrorTrimmed(NPo, pairWiseDistances); // Updating MSE
        std::cout<< mean_error << endl;

        if (abs(prev_error - mean_error) < tolerance) {
            break;
        }
        prev_error = mean_error;
    }
    auto time_finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = (time_finish - time_start);
    std::cout << "Tr ICP Execution time = " << duration.count() << " s" << endl;
    std::cout << "Rotation matrix: " << endl << rotation_matrix << endl;
    std::cout << "Translation matrix: " << endl << translation_matrix << endl;
    saveRegistratoinResultRGB(cvSource, cvTarget, "../TrICP_RGB.txt");
    saveRegistratoinResultRGB(cvSource, cvTarget, "../TrICP.xyz");
    
}

int main() {
  srand(static_cast<unsigned int>(time(nullptr)));
  
 
  string a1 = "../pointCloads/p1_Baby.xyz";
  string a2 = "../pointCloads/p2_Aloe.xyz";
  
  dimention d1 = getDimention(a1);
  dimention d2 = getDimention(a2);

  MatrixXd Mat1(d1.row, d1.col); 
  MatrixXd Mat2(d2.row, d2.col);

  readPointCloud(Mat1,Mat2, a1, a2);
  
  int max_iteration_num = 500;

  //create rotated point cloads
  const double PI = 3.14159265;
  vector<double> degrees = { 0,0,5 * PI / 180.0 };
  //saveRotated(Mat1, degrees, "../pointCloads/p1_Baby_Rotated 5 Over Z axis.xyz");
  addGaussianNoise(Mat1, "../pointCloads/Gaussian noise p1_Baby mean=0 std=0.1 .xyz", 0.0, 0.1);

  // Iterative Closest Point Algorithm
  

  //icp(Mat1, Mat2, max_iteration_num);

  // Trimmed Iterative Closest Point Algorithm
  //tricp(Mat1, Mat2, max_iteration_num); // mat1 is data, mat2 is the model

  
  return 0;
}