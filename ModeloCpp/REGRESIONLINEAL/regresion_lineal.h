#ifndef REGRESIONLINEAL_H
#define REGRESIONLINEAL_H

#include <eigen3/Eigen/Dense>

#include <vector>


class RegresionLineal{

public:
    RegresionLineal()
    {}

    /* A continuación se implementa la función de
     * minimos cuadrados ordinarios como función de costos*/
    float OLS_costo(Eigen::MatrixXd X,Eigen::MatrixXd y,Eigen::MatrixXd Theta);
    std::tuple<Eigen::VectorXd, std::vector<float>> Gradiente(Eigen::MatrixXd X,Eigen::MatrixXd y,Eigen::VectorXd Theta, float alfa, int iterator);
    float R2(Eigen::MatrixXd y,Eigen::MatrixXd y_hat);
};

#endif // REGRESIONLINEAL_H
