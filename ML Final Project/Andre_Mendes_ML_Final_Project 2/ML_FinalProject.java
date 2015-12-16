import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Random;
import java.io.File;
import java.util.Scanner;
import java.lang.*;
import java.net.URL;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.classifiers.*;
import weka.core.converters.LibSVMLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Standardize;
import weka.core.*;
import weka.core.converters.ConverterUtils.*;

@SuppressWarnings({ "deprecation", "unused" })
public class ML_FinalProject {

	@SuppressWarnings({ "unchecked", "rawtypes" })
	public static void main(String[] args) throws Exception {

		// LibSVM models = new LibSVM();
		// Classifier models = new RandomForest();
		Classifier models = new Logistic();
		String modelsName = "AI_Logisitc";
		String dataName = "trainDataAIProject";
		boolean cv = false; // Perform Cross Validation
		boolean rnd = true; // Random permutation in cross validation
		Evaluation validation = null;
		String options;
		FastVector predictions = new FastVector();

		// double[] e = { -15, -13, -11, -9, -7, -5, -3, -1, 0, 1, 3, 5, 7, 9,
		// 11, 13, 15 };
		int b = 0;
		// Read the input file
		BufferedReader datafile = readDataFile(
				"/Users/andremendes/OneDrive/GVGAI-Weka/ARFF Files/" + dataName + ".arff");
		Instances data = new Instances(datafile);
		// Define the class attribute
		data.setClassIndex(data.numAttributes() - 1);
		// Normalize the Data
		Normalize filter_norm = new Normalize();
		filter_norm.setInputFormat(data);
		Instances data_norm = Filter.useFilter(data, filter_norm);
		data_norm.setClassIndex(data_norm.numAttributes() - 1);
		// Split data for Cross Validation
		Instances[][] split = crossValidationSplit(data_norm, 10, rnd);
		Instances[] trainingSplits = split[0];
		Instances[] testingSplits = split[1];

		// for libSVM
		// options = "-C 4 -G 4";
		// models.setOptions(weka.core.Utils.splitOptions(options));

		//check if using cross validation
		if (!cv) {
			validation = classify(models, data_norm, data_norm);

		}

		else
			for (int i = 0; i < trainingSplits.length; i++) {
				validation = classify(models, trainingSplits[i], testingSplits[i]);
				predictions.appendElements(validation.predictions());
			}
		
		//calculate the accuracy of the predictions
		double accuracy = calculateAccuracy(predictions);

		//save the model
		weka.core.SerializationHelper.write("/Users/andremendes/OneDrive/GVGAI-Weka/Models/" + modelsName + ".model",
				models);
		//save the filter
		weka.core.SerializationHelper.write("/Users/andremendes/OneDrive/GVGAI-Weka/Filters/Norm_filter.filter",
				filter_norm);

		//Print the results
		System.out.println(validation.toSummaryString());
		System.out.println(validation.toMatrixString());

		//Print the final accuracy
		System.out.println("Accuracy of " + models.getClass().getSimpleName() + ": " + String.format("%.2f%%", accuracy)
				+ "\n---------------------------------");
	}

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	public static Evaluation classify(Classifier model, Instances trainingSet, Instances testingSet) throws Exception {
		Evaluation evaluation = new Evaluation(trainingSet);

		model.buildClassifier(trainingSet);
		evaluation.evaluateModel(model, testingSet);

		return evaluation;
	}

	public static double calculateAccuracy(@SuppressWarnings("rawtypes") FastVector predictions) {
		double correct = 0;

		for (int i = 0; i < predictions.size(); i++) {
			NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
			if (np.predicted() == np.actual()) {
				correct++;
			}
		}

		return 100 * correct / predictions.size();
	}

	public static Instances[][] crossValidationSplit(Instances data, int numberOfFolds, boolean rnd) {

		Instances[][] split = new Instances[2][numberOfFolds];
		int seed = 1;
		Instances randData;
		Random rand = new Random(seed); // create seeded number generator
		randData = new Instances(data); // create copy of original data
		randData.randomize(rand); // randomize data with number generator

		if (rnd) {
			for (int i = 0; i < numberOfFolds; i++) {
				split[0][i] = randData.trainCV(numberOfFolds, i);
				split[1][i] = randData.testCV(numberOfFolds, i);
			}
		} else
			for (int i = 0; i < numberOfFolds; i++) {
				split[0][i] = data.trainCV(numberOfFolds, i);
				split[1][i] = data.testCV(numberOfFolds, i);
			}
		return split;
	}

	public static String search(double[] e, Instances dataset, LibSVM classifier, int b) throws Exception {

		double accuracy;
		double best_accuracy = 0;
		double bestC = 0;
		double bestGamma = 0;
		double C = 0;
		double gamma = 0;
		String svm_parameter;
		// search for C and gamma
		for (int i = 0; i < e.length; i++) {
			C = Math.pow(2, e[i]);
			for (int j = 0; j < e.length; j++) {
				gamma = Math.pow(2, e[j]);
				accuracy = crossValidation(C, gamma, dataset, classifier, b);
				if (accuracy > best_accuracy) {
					bestC = C;
					bestGamma = gamma;
					best_accuracy = accuracy;
				}
				if (best_accuracy == 100) {
					svm_parameter = "-S 0 -K 2 -B 1 -G " + Double.toString(bestGamma) + " -C " + Double.toString(bestC)
							+ " -B " + Double.toString(b);
					return svm_parameter;
				}

			} // gamma
		} // C}

		svm_parameter = "-S 0 -K 2 -B 1 -G " + Double.toString(bestGamma) + " -C " + Double.toString(bestC) + " -B "
				+ Double.toString(b);
		return svm_parameter;
	}

	@SuppressWarnings({ "unchecked", "rawtypes" })
	private static double crossValidation(Double C, Double gamma, Instances dataset, LibSVM classifier, int b)
			throws Exception {

		String svm_parameter = "-S 0 -K 2 -G " + Double.toString(gamma) + " -C " + Double.toString(C) + " -B "
				+ Double.toString(b);
		classifier.setOptions(weka.core.Utils.splitOptions(svm_parameter));
		Evaluation eval = new Evaluation(dataset);
		eval = classify(classifier, dataset, dataset);
		FastVector predictions = new FastVector();
		predictions.appendElements(eval.predictions());
		double accuracy = calculateAccuracy(predictions);
		return accuracy;
	}

}