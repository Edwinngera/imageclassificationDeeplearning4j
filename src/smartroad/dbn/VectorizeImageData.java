package smartroad.dbn;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.DataSet;

public class VectorizeImageData {

	/**
	 * 
	 * @param width
	 * @param height
	 * @param image
	 * @param labels
	 * @param labelIndex
	 * @return
	 */
	public DataSet getImageDataSet(int width, int height, File image,
			List<String> labels, int labelIndex) {

		// Create a record reader, and initialize it with a FileSplit of the
		// data
		RecordReader reader = new ImageRecordReader(width, height, true, labels);
		try {
			reader.initialize(new FileSplit(image.getAbsoluteFile()));
		} catch (IOException | InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		// Convert record reader to a dataset iterator.
		DataSetIterator iter = new CrackRecordReaderDataSetIterator(reader, 10,
				labelIndex, labels.size());

		// Select next dataset, and scale the data.
		DataSet next = iter.next();
		next.scale();
		return next;
	}

	/**
	 * 
	 * @param width
	 * @param height
	 * @param image
	 * @return
	 */
	public DataSet checkImage(int width, int height, File image) {

		// Create a record reader, and initialize it with a FileSplit of the
		// data
		RecordReader reader = new ImageRecordReader(width, height);
		try {
			reader.initialize(new FileSplit(image.getAbsoluteFile()));
		} catch (IOException | InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		// Convert record reader to a dataset iterator.
		DataSetIterator iter = new CrackRecordReaderDataSetIterator(reader, 10);

		// Select next dataset, and scale the data.
		DataSet next = iter.next();
		next.scale();
		return next;
	}

}