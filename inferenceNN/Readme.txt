Author: Robin Stoffer (robin.stoffer@wur.nl), Date: 27 September 2019
#####################################
General description
#####################################
This is the Readme.txt of the inference_package_MLP13.
The inference_package_MLP13 contains all the scripts and txt-files needed to do inference 
on a manually implemented NN* that eventually will be implemented within our fluid dynamics code microHH.
The inference is done using as inputs 3d-flow fields of three variables (wind velocities u, v, w), 
where a flow field of a certain variables represents a single time step.
Consequently, the input for one time step consists of three 3d-flow snapshots in total (one for each variable).
For optimization, only the time (and memory usage) required to do the inference for one time step is relevant (more details below), 
which is automatically printed to the console.
Large parts of the code only have to be executed once,
 and are therefore not critical for the computational cost involved in running our microHH model.
On my Windows laptop with Intel Core i5 processor and Microsoft Visual Studio with Intel MKL, 
the run time of one time step is approximately 1.2 seconds. 

* = that was previously trained using TensorFlow Estimator API, converted to a frozen graph, 
and subsequently stored in txt-files (i.e. the weights/biases/constants)).
###########################################################
Commands used to compile and run the scripts (on Cartesius)
###########################################################
I (Robin Stoffer) use the following commands to compile the code using gcc/icpc and run the executable on Cartesius:

#Load modules (on Cartesius)
module purge
module load surf-devel
module load 2019
module load imkl/2018.3.222-iimpi-2018b

#Run commands (on Cartesius)
icpc -Wall -o MLP diff_U.h diff_U.cpp Grid.h Grid_test.cpp main_test.cpp Network.h Network.cpp
-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -std=c++14 
-Ofast -xAVX -axCORE-AVX-I,CORE-AVX2,CORE-AVX512 -ipo

or:
g++ -Wall -o MLP diff_U.h diff_U.cpp Grid.h Grid_test.cpp main_test.cpp Network.h Network.cpp 
-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -std=c++14 
-Ofast -march=native

#####################################
Description NN
#####################################
The currently implemented NN is in principle simple: it consists of three separate MLPs that have:
1) an input layer with 285 inputs,
2) a hidden layer with 64 neurons,
3) an output layer with 6 outputs (giving 18 outputs in total).
In the hidden layer, the Leaky Relu function (alpha=0.2) is used as activation function.

Some more detailed notes:
-Although the architecture resembles a MLP, the implemented NN can be seen as a CNN. 
 In the diff_U function (see below), the convolution over a single flow field is implemented manually via 6 for-loops in total (in two separate functions).
 Consequently, the batch size currently used during the inference is 1: for each sample the corresponding input vector is constructed first via the 6 for-loops mentioned.
 By constructing a larger input vector/matrix beforehand (and thus increasing the batch size), the runtime of the script may be reduced further.
 This does come at the expense of a larger memory load.
-The convolution/input vector creation described above is done manually because the procedure is non-standard for our NN.
 The non-standard procedure is needed to ensure that the NN outcomes satisfy the (anti-)symmetry found in the turbulent channel flow case we are looking at,
 and all the required outputs are calculated.
 To be specific, the convolution for our NN is different from a typical CNN in two aspects:
	-The input vector creation consists (conceptually) of two steps: first a block of 5*5*5 grid cells is selected for all three variables, 
	and subsequently for each of the three separate MLPs certain cells are removed from this block.
	This results in a total input size of 285 (5*5*5 + 2*5*4*4) rather than the 375 inputs (3*5*5*5) you would otherwhise expect.
	-The convolution over the flow fields happens in an alternating chessboard-like pattern in the three spatial directions, 
	which is not the same as a typical convolution with stride 2. 
	Furtermore, at the bottom and top of the flow fields the three separate MLPs have to be executed more than at the other vertical levels to get all the required outputs.
	
##################################################################################
Description individual scripts + indication performance-critical parts of the code
##################################################################################
-main_test.cpp: contains the 'main'-function, and creates for in total three time steps the needed variables to do the inference.
Subsequently, for each of these three time steps the diff_U-function is called. 
The run-time involved in the calling of the diff_U function is automatically printed to the screen, 
as this contains all of the performance-critical code.

The main-function itself is not critical for the performance and redundant within our microHH-code.
It simply allows to test the scripts independently from the microHH-code.

-diff_U.h/diff_U.cpp: 
contains the 'diff_U'-function, which does the (non-standard) convolution over the flow snapshots to construct 
the input vector for the NN.
For each grid cell selected in the flow snapshots (which is done in an alternating chessboard-like pattern), 
a surrounding region is selected via the 'select_box'-function.
Subsequently, for each of these input vectors the inference method of the NN class is called that contains all the required computations 
to calculate from the input vector the corresponding 18 outputs.
Finally, these 18 outputs are used to calculated all the required so-called 'tendencies', which are the quantities we are after.

All of this code is critical for the performance, and therefore important to optimize.

-Network.h/Network.cpp: contains the 'Network' class with corresponding member variables and functions, 
and contains all the functionality to manually calculate the output vector from the input vector (defined on purpose outside of the class).
The 'Network' class is used to load and hold all of the MLP variables (weights/biases/constants) upon instantiation, 
which is executed only once as part of the 'main'-function.

The parts of the code related to the calculation of the output vector (functions Inference, output_layer, hidden_layer1) 
are the most critical ones for overall performance in MicroHH. This is especially true for the cblas call in the hidden_layer1 function.
The parts of the code related to the 'Network' class are not relevant to optimize.

-Grid.h/Grid_test.cpp: contains the 'Grid' class with corresponding member variables. 
This class is used to hold all information regarding the grid, and simply contains hard-coded constants.

These two scripts are not relevant to optimize and even redundant within MicroHH. It simply allows to test the scripts independently from microHH.

-variables_MLP13 directory: contains all the variables of the MLP stored as txt-files. 
This is read once upon the instantiation of the 'Network'-class in the 'main'-function.
If you put this directory at another relative location compared to the cpp-scripts, please change the specified path in the 'main'-function accordingly.

########################################
Overview performance-critical parts code
########################################
Network.h/Network.cpp: the functions Inference, output_layer, hidden_layer1.
diff_U.h/diff_U.cpp: all of the code
other parts of the code: not important to optimize for the performance within microHH.

 
 
 
