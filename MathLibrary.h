#pragma once

#include <vector>
#include <Eigen/Core>
#include <Eigen/QR>

class LeastSquareFitting
{
public:
	// Struct encapsulating the polynomial fit's report
	struct polyFitReport
	{
		std::vector<double> coefficients;
		std::vector<double> evaluatedValues;
		double meanError = 0;
		double standardDeviation = 0;
		double maximumError = 0;
		bool isSuccessful;
	};
	// Polynomial interpolation of order k
	void polynomialFit(const std::vector<double>& indices,
		const std::vector<double>& values,
		polyFitReport& report,
		const size_t order);

	void polinomialEvaluation(const std::vector<double>& indices, polyFitReport& report);
};

