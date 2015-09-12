package smartroad.dbn;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.datasets.vectorizer.ImageVectorizer;
import org.deeplearning4j.datasets.vectorizer.Vectorizer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 
 * @author lahiru
 *
 */
public class DBNForSmartRoad {

	private static Logger log = LoggerFactory.getLogger(DBNForSmartRoad.class);

	public static void main(String[] args) throws Exception {

		System.out.println("Start....");
		// Getting the labels
		File rootDir = new File(
				"/home/lahiru/Crack_Project/Testing_images/images");
		// needs to be a list for maintaining order of labels
		List<String> labels = new ArrayList<String>();

		for (File f : rootDir.listFiles()) {
			if (f.isDirectory())
				labels.add(f.getName());
		}

		// Path to training Crack images
		File crackFolder = new File(
				"/home/lahiru/Crack_Project/Testing_images/images/cracks");
		// Path to training non crack images
		File noncrackFolder = new File(
				"/home/lahiru/Crack_Project/Testing_images/images/nonCrackThesh");
		File[] listOfFiles = crackFolder.listFiles();
		DataSet d = null;
		List<DataSet> list = new ArrayList<>();
		for (int i = 0; i < listOfFiles.length; i++) {

			File yourImage = new File(listOfFiles[i].getPath());

			VectorizeImageData v = new VectorizeImageData();
			// yourImage,
			// labels.size(),labels.indexOf(yourImage.getParentFile().getName())
			d = v.getImageDataSet(80, 60, yourImage, labels,
					labels.indexOf(yourImage.getParentFile().getName()));

			list.add(d);
		}

		File[] listOfNonCrackFiles = noncrackFolder.listFiles();

		for (int i = 0; i < listOfNonCrackFiles.length; i++) {

			File yourImage = new File(listOfNonCrackFiles[i].getPath());

			VectorizeImageData v = new VectorizeImageData();
			// yourImage,
			// labels.size(),labels.indexOf(yourImage.getParentFile().getName())
			d = v.getImageDataSet(80, 60, yourImage, labels,
					labels.indexOf(yourImage.getParentFile().getName()));

			list.add(d);
		}

		System.out.println("------------------------------------");
		System.out.println("d.length : " + d.getLabels().length());
		System.out.println("Total Number of images :  " + list.size());
		System.out.println("number of inputs : " + d.numInputs());
		System.out.println("number of outComes : " + d.numOutcomes());
		System.out.println("------------------------------------");

		// Configuring the network.
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.weightInit(WeightInit.NORMALIZED)
				.constrainGradientToUnitNorm(false)
				.iterations(2)
				.activationFunction("sigmoid")
				.layerFactory(LayerFactories.getFactory(RBM.class))
				.lossFunction(
						LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
				.learningRate(1e-1f).nIn(d.numInputs()).nOut(d.numOutcomes())
				.list(3).hiddenLayerSizes(new int[] { 50, 20 })
				.override(new ClassifierOverride(3)).

				build();

		// Adding the configuration to the MultiLayerNetwork.
		MultiLayerNetwork network = new MultiLayerNetwork(conf);

		// Adding the crack data in to a DataSetIterator
		DataSetIterator iter = new ListDataSetIterator(list, 10);
		// Training the Network
		network.fit(iter);
		iter.reset();

		// Getting all the values in the layers.
		List activations = network.feedForward();
		System.out.println(activations.toString());

		// Evaluating the network
		Evaluation eval = new Evaluation();
		while (iter.hasNext()) {

			DataSet d2 = iter.next();

			INDArray predict2 = network.output(d2.getFeatureMatrix());
			eval.eval(d2.getLabels(), predict2);

		}

		System.out.println(network.score());
		log.info(eval.stats());
		System.out.println(eval.accuracy());

		// Save the network as an Object.
		saveNetwork(network);

		// For tesing purposes.
		System.out.println("Test-----");

		// System.out.println(network.activate());
		File crack = new File(
				"/home/lahiru/Crack_Project/Testing_images/Original_Test-Data/11/cracks/A004-F-27km Left Pavement Camera 0012830.jpg");
		File noncrack = new File(
				"/home/lahiru/Crack_Project/Testing_images/images/nonCrackThesh/A004-F-27km Left Pavement Camera 0013060.jpg_THRESH_TRUNC_0.jpg");

		System.out.println("crackImage is a file :" + crack.isFile());
		// Vectorizing the crack image.
		VectorizeImageData v = new VectorizeImageData();
		// DataSet testd = v.checkImage(80, 60, crack);
		DataSet testd = v.getImageDataSet(80, 60, crack, labels,
				labels.indexOf(crack.getParentFile().getName()));
		INDArray predict2 = network.output(testd.getFeatureMatrix());
		System.out.println("crackImage predict : " + predict2.toString());
		System.out.println("-------------------------------------------");
		for (int i = 0; i < predict2.rows(); i++) {
			String actual = network.getLabels().getRow(i).toString().trim();
			String predicted = predict2.getRow(i).toString().trim();
			System.out.println("actual " + actual + " vs predicted "
					+ predicted);
			INDArray rreconstructtest = network.reconstruct(
					testd.getFeatureMatrix(), 4);
			System.out.println("rreconstructtest : "
					+ rreconstructtest.toString());
		}
		System.out.println("-------------------------------------------");

		System.out.println("noncrackImage is a file :" + noncrack.isFile());
		VectorizeImageData v2 = new VectorizeImageData();
		// DataSet testd2 = v2.checkImage(80, 60, noncrack);
		DataSet testd2 = v2.getImageDataSet(80, 60, noncrack, labels,
				labels.indexOf(crack.getParentFile().getName()));
		INDArray predict22 = network.output(testd2.getFeatureMatrix());
		System.out.println("non crackImage predict : " + predict22.toString());
		System.out.println("-------------------------------------------");
		for (int i = 0; i < predict22.rows(); i++) {
			String actual = network.getLabels().getRow(i).toString().trim();
			String predicted = predict22.getRow(i).toString().trim();
			System.out.println("actual " + actual + " vs predicted "
					+ predicted);
			INDArray rreconstructtest = network.reconstruct(
					testd2.getFeatureMatrix(), 4);
			System.out.println("rreconstructtest : "
					+ rreconstructtest.toString());
		}
		System.out.println("-------------------------------------------");
		System.out.println("End-----");

	}

	private static void saveNetwork(MultiLayerNetwork network)
			throws FileNotFoundException, IOException {
		Object obj = network;
		OutputStream file = new FileOutputStream("DBN_network_OriImgs_"
				+ System.currentTimeMillis() + ".dbnet");
		OutputStream buffer = new BufferedOutputStream(file);
		ObjectOutput output = new ObjectOutputStream(buffer);
		try {
			output.writeObject(obj);
		} catch (IOException ex) {
			System.out.println(ex);
		} finally {
			output.close();
			buffer.close();
		}
	}

}
