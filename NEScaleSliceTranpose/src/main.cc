// author : erhan akagunduz

#include <cstdint>

#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Helpers.h"
#include "utils/ImageLoader.h"
#include "utils/Utils.h"

#include <iostream>

int main(int argc,char** argv)
{
	const std::size_t width = 1920U;
	const std::size_t height = 1080U;

	arm_compute::Tensor input{};
	arm_compute::Tensor scaledOutput{};
	arm_compute::Tensor slicedOutput{};
	arm_compute::Tensor transposedOutput{};
	arm_compute::NEScale neScale{};
	arm_compute::NESlice neSlice{};
	arm_compute::NETranspose neTranspose{};

	arm_compute::NEScheduler::set(arm_compute::NEScheduler::Type::CPP);
	arm_compute::IScheduler& scheduler = arm_compute::NEScheduler::get();
	std::uint32_t cpuNum = scheduler.cpu_info().get_cpu_num();
	// std::cout << "CPU count : " << cpuNum << std::endl;
	scheduler.set_num_threads(cpuNum);
	// std::cout << "Thread count : " << scheduler.num_threads() << std::endl;

	// Generate dummy data
	std::uint8_t* const inputData = new std::uint8_t[width * height];
	for (std::size_t h = 0; h < height; h++)
	{
		for (std::size_t w = 0; w < width; w++)
		{
			inputData[h * width + w] = w % 256U;
		}
	}

	/*
	const arm_compute::TensorShape inputTensorShape(width, height);
	arm_compute::TensorInfo inputInfo(inputTensorShape, 1U, arm_compute::DataType::U8);
	input.allocator()->init(inputInfo);

	const arm_compute::TensorShape outputTensorShape(width / 2U, height / 2U);
	arm_compute::TensorInfo outputInfo(outputTensorShape, 1U, arm_compute::DataType::U8);
	input.allocator()->init(outputInfo);
	*/

	arm_compute::TensorInfo inputInfo(width, height, arm_compute::Format::U8);
	arm_compute::TensorInfo scaledOutputInfo(width / 4U, height / 4U, arm_compute::Format::U8);
	arm_compute::TensorInfo slicedOutputInfo(width / 4U, height / 4U, arm_compute::Format::U8);
	arm_compute::TensorInfo transposedOutputInfo(height, width, arm_compute::Format::U8);

	std::cout << "inputInfo " << "dimension(0) : " << inputInfo.dimension(0U) << " dimension(1) : " << inputInfo.dimension(1U) << std::endl;
	std::cout << "scaledOutputInfo " << "dimension(0) : " << scaledOutputInfo.dimension(0U) << " dimension(1) : " << scaledOutputInfo.dimension(1U) << std::endl;

	// Just init allocators, allocation must be done after configuring kernels.
	input.allocator()->init(inputInfo);
	scaledOutput.allocator()->init(scaledOutputInfo);
	slicedOutput.allocator()->init(slicedOutputInfo);
	transposedOutput.allocator()->init(transposedOutputInfo);

	neScale.configure(
			&input,
			&scaledOutput,
			arm_compute::InterpolationPolicy::BILINEAR/*arm_compute::InterpolationPolicy::NEAREST_NEIGHBOR*/,
			arm_compute::BorderMode::REPLICATE
			);

	// determine dummy slice area and configure slice kernel
	arm_compute::Coordinates starts(width / 2U, height / 2U);
	arm_compute::Coordinates stops(width / 2U + width / 4U, height / 2U + height / 4U);
	neSlice.configure(
			&input,
			&slicedOutput,
			starts,
			stops
			);

	neTranspose.configure(
			&input,
			&transposedOutput
			);

	input.allocator()->allocate();
	scaledOutput.allocator()->allocate();
	slicedOutput.allocator()->allocate();
	transposedOutput.allocator()->allocate();

	arm_compute::Window window;
	window.use_tensor_dimensions(input.info()->tensor_shape(), arm_compute::Window::DimY);
	arm_compute::Iterator iter(&input, window);
	// push data into input tensor
	arm_compute::execute_window_loop(
			window,
			[&](const arm_compute::Coordinates& id)
			{
				(void)std::memcpy(
					reinterpret_cast<void*>(iter.ptr()),
					reinterpret_cast<const void*>((const void*)&inputData[id.y() * width]),
					((sizeof(inputData[0U])) * width)
				);
			},
			iter
			);

	// Run neon functions
	neScale.run();
	neSlice.run();
	neTranspose.run();

	printf("Saving input and output tensors as ppm image\n");
	arm_compute::utils::save_to_ppm(input, "Input.ppm");
	arm_compute::utils::save_to_ppm(scaledOutput, "ScaledOutput.ppm");
	arm_compute::utils::save_to_ppm(slicedOutput, "SlicedOutput.ppm");
	arm_compute::utils::save_to_ppm(transposedOutput, "TransposedOutput.ppm");

	return 0;
}
