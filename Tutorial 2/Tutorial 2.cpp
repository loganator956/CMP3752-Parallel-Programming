#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char** argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test.ppm";

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		CImg<unsigned char> image_input(image_filename.c_str());
		CImgDisplay disp_input(image_input, "input");

		//a 3x3 convolution mask implementing an averaging filter
		std::vector<float> convolution_mask = { 1.f / 9, 1.f / 9, 1.f / 9,
												1.f / 9, 1.f / 9, 1.f / 9,
												1.f / 9, 1.f / 9, 1.f / 9 };

		//Part 3 - host operations
		//3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);

		//3.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Part 4 - device operations

		std::vector<int> cumulative_histogram(256 * image_input.spectrum(), 0);
		std::vector<int> normalised_histogram(256 * image_input.spectrum(), 0);
		std::vector<int> max_hist(256 * image_input.spectrum(), 0);

		//device - buffers
		cl::Buffer dev_cumulative_histogram(context, CL_MEM_READ_WRITE, cumulative_histogram.size() * sizeof(int));
		cl::Buffer dev_max_histogram(context, CL_MEM_READ_WRITE, max_hist.size() * sizeof(int));

		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size()); //should be the same as input image

				//4.1 Copy images to device memory
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);

		//  STEP 1 :: Generate Intensity Histogram
		//		buffers
		std::vector<int> intensity_histogram(256 * image_input.spectrum(), 0);
		cl::Buffer dev_intensity_histogram(context, CL_MEM_READ_WRITE, intensity_histogram.size() * sizeof(int));
		//		kernel
		cl::Kernel ihistKernel = cl::Kernel(program, "histogram255");
		ihistKernel.setArg(0, dev_image_input);
		ihistKernel.setArg(1, dev_intensity_histogram);
		queue.enqueueNDRangeKernel(ihistKernel, cl::NullRange, cl::NDRange(image_input.width(), image_input.height(), image_input.spectrum()), cl::NullRange);
		//		read
		queue.enqueueReadBuffer(dev_intensity_histogram, CL_TRUE, 0, intensity_histogram.size() * sizeof(int), &intensity_histogram[0]);
		std::cout << intensity_histogram << std::endl;

		//  STEP 2 :: Calculate cumulative histogram
		queue.enqueueWriteBuffer(dev_intensity_histogram, CL_TRUE, 0, intensity_histogram.size() * sizeof(int), &intensity_histogram[0]);
		
		cl::Kernel cumulativeHistKernel = cl::Kernel(program, "scan_add");
		cumulativeHistKernel.setArg(0, dev_intensity_histogram);
		cumulativeHistKernel.setArg(1, dev_cumulative_histogram);
		cumulativeHistKernel.setArg(2, cl::Local(intensity_histogram.size() * sizeof(int)));
		cumulativeHistKernel.setArg(3, cl::Local(intensity_histogram.size() * sizeof(int)));
		
		for (int i = 0; i < image_input.spectrum(); i++)
			queue.enqueueNDRangeKernel(cumulativeHistKernel, cl::NDRange(256 * i), cl::NDRange(256));
		//  STEP 3 :: Normalise histogram
		//		find max
		queue.enqueueReadBuffer(dev_cumulative_histogram, CL_TRUE, 0, cumulative_histogram.size() * sizeof(int), &cumulative_histogram[0]);
		int max = cumulative_histogram[cumulative_histogram.size() - 1];
		for (int i = 0; i < image_input.spectrum(); i++)
		{
			int v = cumulative_histogram[(256 * (i + 1)) - 1];
			if (v > max)
				max = v;
		}
		//		devices
		cl::Buffer dev_divideby(context, CL_MEM_READ_WRITE, sizeof(int));
		cl::Buffer dev_normalised_histogram(context, CL_MEM_READ_WRITE, cumulative_histogram.size() * sizeof(int));
		queue.enqueueWriteBuffer(dev_divideby, CL_TRUE, 0, sizeof(int), &max);
		//		kernel
		cl::Kernel normalise = cl::Kernel(program, "divide");
		normalise.setArg(0, dev_cumulative_histogram);
		normalise.setArg(1, dev_normalised_histogram);
		normalise.setArg(2, dev_divideby);
		queue.enqueueNDRangeKernel(normalise, cl::NullRange, cl::NDRange(cumulative_histogram.size()));

		//queue.enqueueReadBuffer(dev_normalised_histogram, CL_TRUE, 0, cumulative_histogram.size() * sizeof(int), &normalised_histogram[0]);
		//std::cout << normalised_histogram << std::endl;
		
		// TODO: back-projection

		//  STEP 4 :: Back-projection using lut

		cl::Kernel backprojection = cl::Kernel(program, "project");
		backprojection.setArg(0, dev_image_input);
		backprojection.setArg(1, dev_normalised_histogram);
		backprojection.setArg(2, dev_image_output);
		
		queue.enqueueNDRangeKernel(backprojection, cl::NullRange, cl::NDRange(image_input.width(), image_input.height(), image_input.spectrum()), cl::NullRange);





		vector<unsigned char> output_buffer(image_input.size());
		//4.3 Copy the result from device to host
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(output_image, "output");

		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
			disp_input.wait(1);
			disp_output.wait(1);
		}

	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
