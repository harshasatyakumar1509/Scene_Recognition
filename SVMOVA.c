#include <stdlib.h> 
#include <stdio.h> 
#include <math.h> 
#include <time.h>
#include <sys/time.h>
#include <mpi.h>
#include <string.h>
#include <ctype.h>

#define WEIGHTSIZE 491
#define OUTPUTCLASS 3
#define TRAINNUMBER 500
#define TRAINPARAMETERS 490
#define TESTNUMBER 176
/*#define LAMBDA 0.1*/

void SVMWeightCalculation(int output_number, float temp_weight_row[491], float train_data[500][490], int train_label[500], float lambda);
int SVMTesting(float WeightMatrix[OUTPUTCLASS][491], int start_number, int end_number);

int main(int argc, char *argv[])
{
	int myid, numprocs;
	int i,j;
	int local_accuracy = 0;
	int final_accuracy = 0;
	int start_number = 0;
	int end_number = 0;
	float accuracy_percentage = 0;
	float lambda = 0.4;
	int output_number = 0;
	float WeightMatrix[3][491];
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &myid);
	MPI_Comm_size (MPI_COMM_WORLD, &numprocs);
	
	int sizearray[25] = {0};
	int donearray[25] = {0};
	int sizearraytestdata[25] = {0};
	int donearraytestdata[25] = {0};
	float train_data[500][490];
	int train_label[500] = {0};
	float test_data[176][490];
	int test_label[176];
	int temp_test_label[176];
	float temp_weight_row[491] = {0};
	
	int sizecount[25] = {0};
	int displacement[25] = {0};
	int average = 3/numprocs;
	int leftovers = 3%numprocs;
	int outputarray[3] = {4,11,13};
	int average_testdata = TESTNUMBER/numprocs;
	int leftovers_testdata = TESTNUMBER%numprocs;
	int procsused = 0;
	
	//Read The Training Data
	FILE *fp1 = fopen("images_train.txt","r");
	for(i=0;i<500;i++){
		for(j=0;j<490;j++){
			fscanf(fp1, "%f ", &train_data[i][j]);
		}
	}
	fclose(fp1);
	
	//Read The Training Labels
	FILE *fp2 = fopen("labels_train.txt","r");
	for(i=0;i<500;i++){
		fscanf(fp2, "%d ", &train_label[i]);
	}
	fclose(fp2);
	
	//Read The Test Data
	FILE *fp3 = fopen("images_test.txt","r");
	for(i=0;i<176;i++){
		for(j=0;j<490;j++){
			fscanf(fp3, "%f ", &test_data[i][j]);
		}
	}
	fclose(fp3);
	
	//Read The Test Labels
	FILE *fp4 = fopen("labels_test.txt","r");
	for(i=0;i<176;i++){
		fscanf(fp4, "%d ", &test_label[i]);
	}
	fclose(fp4);
	//For calculating number of elements for which SVM will find the weight matrix
	for (i=0; i<numprocs; i++)
	{
		if(i<leftovers)
		{
			sizearray[i] = average + 1;
		}
		else
		{
			sizearray[i] = average;			
		}
	}
	//For calculating starting number for each processor
	procsused = (numprocs > 3) ? 3 : numprocs;
	printf("%d", procsused);
		for (i=0; i<procsused; i++)
		{
			if(i==0)
			{
				donearray[i] = 0;
			}	
			else
			{
				for(j=0;j<i;j++)
				{
					donearray[i] = donearray[i] + sizearray[j];
				}	
				
			}
		}
	//Calculating the number of test data handled by each processor
	for (i=0; i<numprocs; i++)
	{
		if(i<leftovers_testdata)
		{
			sizearraytestdata[i] = average_testdata + 1;
		}
		else
		{
			sizearraytestdata[i] = average_testdata;			
		}
	}
	//For calculating starting number for testdata handled by each processor
	for (i=0; i<numprocs; i++)
	{
		if(i==0)
		{
			donearraytestdata[i] = 0;
		}	
		else
		{
			for(j=0;j<i;j++)
			{
				donearraytestdata[i] = donearraytestdata[i] + sizearraytestdata[j];
			}	
			
		}
	}
	
	float temp_weight[sizearray[myid]][491];
	//Calculating Weight for respective number of output labels. From donearray[myid] to donearray[myid]+sizearray[myid]
	for (output_number = donearray[myid]; output_number< donearray[myid]+sizearray[myid]; output_number++)
	{
		//printf("\nOutput Class : %d", outputarray[output_number - donearray[myid]]);
		SVMWeightCalculation(outputarray[output_number], temp_weight_row, train_data, train_label, lambda);
		for (j = 0; j < WEIGHTSIZE; j++)
		{
			temp_weight[output_number-donearray[myid]][j] = temp_weight_row[j];
		}
	}
	
	for(i=0;i<numprocs;i++)
	{
		sizecount[i] = sizearray[i]*491; 
		displacement[i] = donearray[i]*491; 
	}
	
	MPI_Gatherv(temp_weight, (sizearray[myid]*491), MPI_FLOAT, WeightMatrix, sizecount , displacement, MPI_FLOAT, 0, MPI_COMM_WORLD);
	
	start_number = donearraytestdata[myid];
	end_number = donearraytestdata[myid] + sizearraytestdata[myid] - 1;
	//Calculating accuracy for specific data points on each processor
	
	
	MPI_Bcast(WeightMatrix, OUTPUTCLASS*491, MPI_FLOAT, 0, MPI_COMM_WORLD);
	
	local_accuracy = SVMTesting(WeightMatrix, start_number, end_number);
	//printf("\n Local Accuracy at %d is %d", myid, local_accuracy);
	MPI_Reduce(&local_accuracy, &final_accuracy, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	if(myid == 0){
		//printf("\n Final Accuracy is : %d", final_accuracy);
		accuracy_percentage = (final_accuracy/(double)(176));
		accuracy_percentage = accuracy_percentage*100;
		printf("\n Accuracy Percentage is : %f", accuracy_percentage);
	}
	
	
	MPI_Finalize();
	return 0;
}

/*SVM Training, Calculate weight matrix*/
void SVMWeightCalculation(int output_number, float temp_weight_row[491], float train_data[500][490], int train_label[500], float lambda)
{
	int i,j,k,l;
	float div_factor = 0.0000;
	int temp_train_label[500];
	float sum = 0;
	float activation = 0;

	for(i=0; i<WEIGHTSIZE; i++)
	{
		temp_weight_row[i] = 0;
	}
	for(i=0; i<TRAINNUMBER; i++)
	{
		if(train_label[i] == output_number) {
			temp_train_label[i] = 1;			
		}
		else {
			temp_train_label[i] = -1;			
		}
	}

	for(i=0; i<500; i++){
		sum = 0;

		for(j=0; j<WEIGHTSIZE-1; j++){
			sum = sum + train_data[i][j]*temp_weight_row[j];
		}
		sum = sum + temp_weight_row[WEIGHTSIZE-1];
		activation = temp_train_label[i]*sum;
		
		div_factor = 1/((double)(i+1));
		if(activation <= 1){
			for(k = 0; k < WEIGHTSIZE; k++){
				if(k < WEIGHTSIZE-1){
					temp_weight_row[k] = temp_weight_row[k] + (div_factor*temp_train_label[i]*train_data[i][k]);
				}
				else{
					temp_weight_row[k] = temp_weight_row[k] + (div_factor*temp_train_label[i]);
				}
			}
		}
		else{
			for(k = 0; k < WEIGHTSIZE; k++){
				temp_weight_row[k] = (1-div_factor*lambda)*temp_weight_row[k];
			}
		}
	}

}

//SVM Testing, calculating the local accuracy
int SVMTesting(float WeightMatrix[3][491], int start_number, int end_number)
{
	int i,j,k,l;
	int testoutputs[3] = {4, 11, 13};
	int predictedoutput = 0;
	int max_value = -10000;
	float testimageactivation[3] = {0};
	int local_accuracy = 0;
	float test_data[176][490];
	int test_label[176];
	
	//Read The Test Data
	FILE *fp3 = fopen("images_test.txt","r");
	for(i=0;i<176;i++){
		for(j=0;j<490;j++){
			fscanf(fp3, "%f ", &test_data[i][j]);
		}
	}
	fclose(fp3);
	
	//Read The Test Labels
	FILE *fp4 = fopen("labels_test.txt","r");
	for(i=0;i<176;i++){
		fscanf(fp4, "%d ", &test_label[i]);
	}
	fclose(fp4);

	for(i=start_number;i<end_number + 1;i++){
		max_value = -10000;
		for(l=0;l<3;l++){
			testimageactivation[l] = 0;
		}
		j = 0;
		for(j=0;j<OUTPUTCLASS;j++){
			for(k=0; k<WEIGHTSIZE-1; k++){
				testimageactivation[j] = testimageactivation[j] + test_data[i][k]*WeightMatrix[j][k];
			}
			testimageactivation[j] = testimageactivation[j] + WeightMatrix[j][WEIGHTSIZE-1];
		}
		
		predictedoutput = 0;
		l=0;
		while(l<OUTPUTCLASS){
			if(testimageactivation[l] > max_value){
				max_value = testimageactivation[l];
				predictedoutput = testoutputs[l];
			}
			else{
				if(predictedoutput==0)
				{
					predictedoutput = testoutputs[l];
				}
			}
			l = l + 1;
		}
		//printf("\n%d Actual Output : %d, Predicted Output : %d", i, test_label[i], predictedoutput);
		if(predictedoutput == test_label[i]){
			local_accuracy = local_accuracy + 1;
		}
	}
	return local_accuracy;
}