#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <time.h>
#include <iomanip>
#include <thread>

using namespace cv;

	Mat doKmeans(Mat &input, Mat &centers, int flags, TermCriteria criteria, int clusterCount){
		Mat src = input;
		Mat samples(src.rows * src.cols, 3, CV_32F);
		for (int y = 0; y < src.rows; y++){
			for (int x = 0; x < src.cols; x++){
				for (int z = 0; z < 3; z++){
					samples.at<float>(y + x*src.rows, z) = src.at<Vec3b>(y, x)[z];
				}
			}
		}
		
		Mat labels;
		int attempts = 1;
		TermCriteria termCriteria;
		kmeans(samples, clusterCount, labels, criteria, attempts, flags, centers);

		Mat new_image(src.size(), src.type());
		for (int y = 0; y < src.rows; y++)
		for (int x = 0; x < src.cols; x++)
		{
			int cluster_idx = labels.at<int>(y + x*src.rows, 0);
			new_image.at<Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
			new_image.at<Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
			new_image.at<Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
		}
		return new_image;
	}

	/*
	* Code from http://jepsonsblog.blogspot.de/2012/10/overlay-transparent-image-in-opencv.html
	*/
	void overlayImage(const cv::Mat &background, const cv::Mat &foreground,cv::Mat &output, cv::Point2i location)
	{
		background.copyTo(output);
		// start at the row indicated by location, or at row 0 if location.y is negative.
		for (int y = std::max(location.y, 0); y < background.rows; ++y)
		{
			int fY = y - location.y; // because of the translation

			// we are done of we have processed all rows of the foreground image.
			if (fY >= foreground.rows)
				break;

			// start at the column indicated by location, 

			// or at column 0 if location.x is negative.
			for (int x = std::max(location.x, 0); x < background.cols; ++x)
			{
				int fX = x - location.x; // because of the translation.

				// we are done with this row if the column is outside of the foreground image.
				if (fX >= foreground.cols)
					break;

				// determine the opacity of the foregrond pixel, using its fourth (alpha) channel.
				double opacity =
					((double)foreground.data[fY * foreground.step + fX * foreground.channels() + 3])

					/ 255.;


				// and now combine the background and foreground pixel, using the opacity, 

				// but only if opacity > 0.
				for (int c = 0; opacity > 0 && c < output.channels(); ++c)
				{
					unsigned char foregroundPx =
						foreground.data[fY * foreground.step + fX * foreground.channels() + c];
					unsigned char backgroundPx =
						background.data[y * background.step + x * background.channels() + c];
					output.data[y*output.step + output.channels()*x + c] =
						backgroundPx * (1. - opacity) + foregroundPx * opacity;
				}
			}
		}
	}

	Mat removeBlackToTransparent(const Mat &input, int lowerThresh, int upperThresh){
		Mat img = input;
		Mat dst;//(src.rows,src.cols,CV_8UC4);
		Mat tmp, alpha;

		Mat out;
		cvtColor(img, tmp, CV_BGR2GRAY);
		threshold(tmp, alpha, lowerThresh, upperThresh, THRESH_BINARY);

		Mat rgb[3];
		split(img, rgb);

		Mat rgba[4] = { rgb[0], rgb[1], rgb[2], alpha };
		merge(rgba, 4, dst);

		// Convert image to correct output format 
		dst.convertTo(out, CV_8UC3);
		return out;
	}


	/*
	* Calculates the LaPlace outline for the overlay for the input image.
	*
	*/
	Mat doLaPlaceFilter(Mat input){
		Mat laPlaceFiltered;

		//weight
		double alpha = 0.8;
		double beta = (1.0 - alpha);
		// Laplace settings
		int kernel_size = 5;
		int scale = 1;
		int delta = 0;
		int ddepth = CV_32F;

		//Threshold settings
		int maxValue = 255;
		double thresholdValue = 254;

		//Blud settings
		int blurRadius = 3;

		//Apply LaPlace
		Laplacian(input, laPlaceFiltered, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);

		//process filter laplace to get the outline
		threshold(laPlaceFiltered, laPlaceFiltered, thresholdValue, maxValue, 0);

		return laPlaceFiltered;
	}


	int fadeIntLinear(int number, int from, int to, int startFrame, int endFrame, int currentFrame){
		if (currentFrame < startFrame){
			return number;
		}
		else if (currentFrame >= endFrame){
			return to;
		}
		else {
			int range = to - from;

			// Convert ints to doubles so the compiler knows to perform a floatingpoint operation
			double cf = currentFrame;
			double ef = endFrame;
			// Calc percentage
			double perc = cf / ef;

			return from + range * perc;
		}
	}

	double fadeDoubleLinear(double number, double from, double to, int startFrame, int endFrame, int currentFrame){
		if (currentFrame < startFrame){
			return number;
		}
		else if (currentFrame >= endFrame){
			return to;
		}
		else {
			double range = to - from;
			
			// Convert ints to doubles so the compiler knows to perform a floatingpoint operation
			double cf = currentFrame;
			double ef = endFrame;
			// Calc percentage
			double perc = cf / ef;

			return from + range * perc;
		}
	}


	void processFrame(Mat &input, int frameNumber){
		int frameStartTime = 0, frameEndTime = 0;
		frameStartTime = time(NULL);
		Mat frameSrc = input;
		Mat frameProcessed;
		Mat frameLaPlace;
		Mat frameKMeans;


		// ============================================= PREPROCESS CALC =================================================================
		medianBlur(frameSrc, frameProcessed, 3);


		// ============================================= LAPLACE CALC ====================================================================

		// This version of the laplace filter is created from the preprocessed version
		// of the source image. it will be used later in the process
		frameLaPlace = doLaPlaceFilter(frameProcessed);
		// Create the outline
		frameLaPlace = removeBlackToTransparent(frameLaPlace, 250, 255);




		// ============================================= K_MEANS CALC ====================================================================
		Mat centers;
		int flags = KMEANS_PP_CENTERS;
		int clusterCount = 8;

		// ------------------------------------------------------------------------
		// increase clusters
		clusterCount = fadeIntLinear(clusterCount, 8, 18, 268, 318, frameNumber);
		clusterCount = fadeIntLinear(clusterCount, 8, 8, 319, 319, frameNumber);

		clusterCount = fadeIntLinear(clusterCount, 4, 4, 965, 965, frameNumber);
		// ------------------------------------------------------------------------

		TermCriteria criteria = TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001);
		frameKMeans = doKmeans(frameProcessed, centers, flags, criteria, clusterCount);




		// ============================================= K_MEANS OVERLAY ====================================================================
		Mat tmp_KMeans_Src;

		// ------------------------------------------------------------------------
		// Fade in K-Means
		double alphaKMeans = 0.0;
		alphaKMeans = fadeDoubleLinear(alphaKMeans, 0.0, 0.8, 0, 187, frameNumber);
		alphaKMeans = fadeDoubleLinear(alphaKMeans, 0.8, 0.0, 319, 363, frameNumber);
		alphaKMeans = fadeDoubleLinear(alphaKMeans, 0.0, 0.8, 422, 422, frameNumber);

		alphaKMeans = fadeDoubleLinear(alphaKMeans, 0.0, 0.5, 965, 1137, frameNumber);
		alphaKMeans = fadeDoubleLinear(alphaKMeans, 0.5, 0.8, 1138, 1390, frameNumber);
		// ------------------------------------------------------------------------

		addWeighted(frameKMeans, alphaKMeans, frameSrc, 1 - alphaKMeans, 0.4, tmp_KMeans_Src);




		// ============================================= LAPLACE OVERLAY ====================================================================
		Mat tmp_LaPlace_KMeans_Src;
		// Use this algorithm because addWeighted does not work with transparency
		overlayImage(tmp_KMeans_Src, frameLaPlace, tmp_LaPlace_KMeans_Src, Point2i(0, 0));

		// ------------------------------------------------------------------------
		// Fade in the outline
		double alphaLaPlace = 0.0;
		alphaLaPlace = fadeDoubleLinear(alphaLaPlace, 0.0, 0.9, 188, 240, frameNumber);
		alphaLaPlace = fadeDoubleLinear(alphaLaPlace, 0.9, 0.0, 241, 267, frameNumber);
		alphaLaPlace = fadeDoubleLinear(alphaLaPlace, 0.0, 0.9, 319, 319, frameNumber);
		alphaLaPlace = fadeDoubleLinear(alphaLaPlace, 0.9, 0.1, 340, 363, frameNumber);
		alphaLaPlace = fadeDoubleLinear(alphaLaPlace, 0.1, 0.9, 422, 422, frameNumber);

		alphaLaPlace = fadeDoubleLinear(alphaLaPlace, 0.0, 0.7, 965, 1137, frameNumber);
		alphaLaPlace = fadeDoubleLinear(alphaLaPlace, 0.0, 0.9, 1138, 1390, frameNumber);
		// ------------------------------------------------------------------------

		addWeighted(tmp_LaPlace_KMeans_Src, alphaLaPlace, tmp_KMeans_Src, 1 - alphaLaPlace, 0.4, frameProcessed);

		// ============================================= SAVE IMAGE ====================================================================
		std::stringstream outName;
		outName << "out/" << frameNumber << ".png";
		imwrite(outName.str(), frameProcessed);

	}

	

	/*
	* Main Method
	* Calls all the other methods
	*/
	int main(int argc, char** argv){
		std::cout << "Innovative Ansaetze in der Medieninformatik - Projekt v1.2 \n";
		int startTime = time(NULL), endTime, totalTime, renderTimePassed, currentTime;
		const string resourceBasePath = "raw/";
		const string videoName = "SequenceFull.avi";
		const string outVideoName = videoName;

		int currentFrame = 0;
		VideoWriter writer;

		VideoCapture cap(resourceBasePath + videoName);

		//Crossframe members
		Mat lastClusters;

		// Setup Video writer
		double fps = cap.get(CV_CAP_PROP_FPS);
		int frameCount = cap.get(CV_CAP_PROP_FRAME_COUNT);
		Size videoSize(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
		// Try to open writer
		writer.open(outVideoName, cap.get(CV_CAP_PROP_FOURCC), fps, videoSize,true);

		//Check if opening the 
		if (!cap.isOpened()){
			std::cout << "[Main] " << "Cannot open the video file" << "\n";
			return 0;
		}

		//Check if writer works
		if (!writer.isOpened()){
			std::cout << "[Main] " << "Cannot open videowriter. Retrying with default settings" << "\n";
			writer.open(outVideoName, CV_FOURCC_DEFAULT, fps, videoSize, true);
		}
			

		std::cout << "[Main] " << "Starting with video "
			<< videoName << " Size: "
			<< cap.get(CV_CAP_PROP_FRAME_WIDTH) << "x" << cap.get(CV_CAP_PROP_FRAME_HEIGHT)
			<< " Frame count: " << frameCount
			<< " Fps: " << cap.get(CV_CAP_PROP_FPS)
			<< "\n";

		bool frameAvailiable = true;
		// Loop through all the frames

		std::thread t[4];
		Mat inFrames[4];
		bool t_running[4];

		while (frameAvailiable){

			frameAvailiable = cap.read(inFrames[0]);
			if (frameAvailiable && currentFrame < frameCount){
				t[0] = std::thread(processFrame, inFrames[0], currentFrame);
				currentFrame++;
				t_running[0] = true;
			}

			frameAvailiable = cap.read(inFrames[1]);
			if (frameAvailiable && currentFrame < frameCount){
				t[1] = std::thread(processFrame, inFrames[1], currentFrame );
				currentFrame++;
				t_running[1] = true;
			}

			frameAvailiable = cap.read(inFrames[2]);
			if (frameAvailiable && currentFrame < frameCount){
				t[2] = std::thread(processFrame, inFrames[2], currentFrame);
				currentFrame++;
				t_running[2] = true;
			}

			frameAvailiable = cap.read(inFrames[3]);
			if (frameAvailiable && currentFrame < frameCount){
				t[3] = std::thread(processFrame, inFrames[3], currentFrame);
				currentFrame++;
				t_running[3] = true;
			}


			// Wait for threads to finish
			if (t_running[0]){
				t_running[0] = false;
				try{
					t[0].join();
				}
				catch (int e){
					std::cout << "[Main] " << "Unable to join thread 0: " << e << "\n";
				}
			}
			
			if (t_running[1]){
				t_running[1] = false;
				try{
					t[1].join();
				}
				catch (int e){
					std::cout << "[Main] " << "Unable to join thread 1: " << e << "\n";
				}
			}

			if (t_running[2]){
				t_running[2] = false;
				try{
					t[2].join();
				}
				catch (int e){
					std::cout << "[Main] " << "Unable to join thread 2: " << e << "\n";
				}
			}

			if (t_running[3]){
				t_running[3] = false;
				try{
					t[3].join();
				}
				catch (int e){
					std::cout << "[Main] " << "Unable to join thread 3: " << e << "\n";
				}
			}

			// Print current time
			currentTime = time(NULL);
			renderTimePassed = currentTime - startTime;
			std::cout << "[Main] " << "Time passed: " << renderTimePassed << "s" << " Frames processed: " << currentFrame << "/" << frameCount <<"\n";
		}

		//Build video
		std::cout << "[Main] " << "Building Video file..." << "\n";
		for (int i = 0; i < frameCount; i++){			
			std::stringstream outName;
			outName << "out/" << i << ".png";
			Mat frame = imread(outName.str() , 1);
			if (writer.isOpened()){
				writer.write(frame);
			}
			else {
				std::cout << "[Main] " << "Unable to write frame " << i << "\n";
			}
		}

		//release cap and writer after successful operation
		cap.release();
		writer.release();


		// Calculate run
		endTime = time(NULL);
		totalTime = endTime - startTime;
		std::cout << "[Main] " << "Runtime: " << totalTime << " seconds.\n";
		waitKey(0);
	}