// This script is copied from Steven C. Shaffer
// My first neural network
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

//Contants
const int NUMINPUTNODES = 2;
const int NUMHIDDENNODES = 2;
const int NUMOUTPUTNODES = 1;
const int NUMNODES = NUMINPUTNODES + NUMHIDDENNODES + NUMOUTPUTNODES;
const int ARRAYSIZE = NUMNODES + 1;
const int MAXITERATIONS = 128;
const double E = 2.71828;
const double LEARNINGRATE = 0.2;

//Function prototypes
void initialize(double[][ARRAYSIZE],double[],double[],double[]);
void connectNodes(double[][ARRAYSIZE],double[]);
void trainingExample(double[],double[]);
void activateNetwork(double[][ARRAYSIZE],double[],double[]);
double updateWeights(double[][ARRAYSIZE],double[],double[],double[]);
void displayNetwork(double[],double);

int main() {
	printf("Neural Network Prototype Program\n");

	double weights[ARRAYSIZE][ARRAYSIZE];
	double values[ARRAYSIZE];
	double expectedValues[ARRAYSIZE];
	double thresholds[ARRAYSIZE];

	initialize(weights, values, expectedValues, thresholds);	
	connectNodes(weights, thresholds);

	int counter = 0;
	while (counter < MAXITERATIONS)
	{
		trainingExample(values,expectedValues);
		activateNetwork(weights, values, thresholds);
		
		double sumOfSquaredErrors = updateWeights(weights, values, expectedValues, thresholds);
		displayNetwork(values, sumOfSquaredErrors);

		counter++;
	}
	return 0;
}

void initialize(double weights[][ARRAYSIZE], double values[], double expectedValues[], double thresholds[])
	{
		for(int x=0;x<=NUMNODES;x++)
		{
			values[x]=0.0;
			expectedValues[x]=0.0;
			thresholds[x]=0.0;
			
			for(int y=0;y<=NUMNODES;y++)
			{
				weights[x][y]=0.0;
			}
		}
	}

void connectNodes(double weights[][ARRAYSIZE],double thresholds[])
{
	for(int x=1;x<=NUMNODES;x++)
	{
		for(int y=1;y<=NUMNODES;y++)
		{
			weights[x][y]=(rand() % 200) / 100.0;
		}
	}

thresholds[3]=rand() / (double)rand();
thresholds[4]=rand() / (double)rand();
thresholds[5]=rand() / (double)rand();

printf("%f %f %f %f %f %f\n%f %f %f\n", weights[1][3], weights[1][4], weights[2][3], \
	weights[2][4], weights[3][5], weights[4][5], thresholds[3], thresholds[4], thresholds[5]);

}

void trainingExample(double values[], double expectedValues[])
{
	static int counter = 0;
	
	switch (counter % 4)
	{
		case 0:
			values[1]=1;
			values[2]=1;
			expectedValues[5]=0;
			break;
		case 1:
			values[1]=0;
			values[2]=1;
			expectedValues[5]=1;
			break;
		case 2:
			values[1]=1;
			values[2]=0;
			expectedValues[5]=1;
			break;
		case 3:
			values[1]=0;
			values[2]=0;
			expectedValues[5]=1;
			break;
		}

	counter++;
}

void activateNetwork(double weights[][ARRAYSIZE], double values[], double thresholds[])
{
	// For each hidden node
	for(int h=1+NUMINPUTNODES;h<1+NUMINPUTNODES+NUMHIDDENNODES;h++)
	{
		double weightedInput=0.0;

		// Add up the weighted Input
		for(int i=1;i<1+NUMINPUTNODES;i++)
		{
			weightedInput+=weights[i][h]*values[i];
		}

		// Handle the Thresholds
		weightedInput += (1*thresholds[h]);

		values[h] = 1.0 / (1.0 + pow(E,weightedInput));
	}

	// For each output node
	for(int o=1 + NUMINPUTNODES + NUMHIDDENNODES; o<1 + NUMNODES;o++)
	{
		double weightedInput = 0.0;
		
		for(int h=1 + NUMINPUTNODES;h<1 + NUMINPUTNODES + NUMHIDDENNODES;h++)
		{
			weightedInput += weights[h][o]*values[h];
		}

		// Handle the Thresholds
		weightedInput += (-1 * thresholds[o]);
		values[o] = 1.0 / (1.0 + pow(E,weightedInput));
	}
}

double updateWeights(double weights[][ARRAYSIZE], double values[], double expectedValues[], double thresholds[])
{
	double sumOfSquaredErrors = 0.0;
	
	for(int o=1 + NUMINPUTNODES + NUMHIDDENNODES; o<1 + NUMNODES;o++)
	{
		double absoluteError = expectedValues[o] - values[o];
		sumOfSquaredErrors += pow(absoluteError,2);
		double outputErrorGradient = values[o] * (1.0 - values[o]) * absoluteError;

		// Update each weighting from the hidden layer
		for(int h=1 + NUMINPUTNODES;h<1 + NUMHIDDENNODES;h++)
		{
			double delta = LEARNINGRATE * values[h] * outputErrorGradient;
			weights[h][o] += delta;
			double hiddenErrorGradient = values [h] * (1 - values[h]) * outputErrorGradient \
				* weights[h][o];
			
			for(int i=1;i<1+NUMINPUTNODES;i++)
			{
				double delta = LEARNINGRATE * values[i] * hiddenErrorGradient;
				weights[i][h] += delta;
			}

			double thresholdDelta = LEARNINGRATE * -1 * hiddenErrorGradient;
			thresholds[h] += thresholdDelta;
		}

		// Update each Weighting for the Theta
		double delta = LEARNINGRATE * -1 * outputErrorGradient;
		thresholds[o] += delta;
	}
	return sumOfSquaredErrors;
}

void displayNetwork(double values[], double sumOfSquaredErrors)
{
	static int counter = 0;
	
	if ((counter % 4) == 0)
		printf("--------------------------------------------------------\n");
		printf("%8.4f|",values[1]);
		printf("%8.4f|",values[2]);
		printf("%8.4f|",values[5]);
		printf("	err:%8.5f\n",sumOfSquaredErrors);
		counter++;
}	
