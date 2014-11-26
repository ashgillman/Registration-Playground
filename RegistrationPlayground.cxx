#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegistrationMethod.h"
#include "itkTranslationTransform.h"
#include "itkMutualInformationImageToImageMetric.h"
#include "itkGradientDescentOptimizer.h"
#include "itkNormalizeImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkCheckerBoardImageFilter.h"
#include "itkSubtractImageFilter.h"
#include <iostream>

// IO filenames
const char* FIXED_FILE = "data/B006_LFOV_N4.nii.gz";
const char* MOVING_FILE = "data/B006_PLAN_CT.nii.gz";
const char* OUT_FILE = "data/out.nii.gz";
const char* DIFF_FILE = "data/dif.nii.gz";

// Image Types
const unsigned int DIM = 3;
typedef float PixelType;
typedef itk::Image<PixelType, DIM> ImageType;
typedef ImageType FixedImageType;
typedef ImageType MovingImageType;
typedef itk::ImageFileReader<ImageType> ReaderType;
typedef itk::ImageFileWriter<ImageType> WriterType;
typedef itk::ImageSource<ImageType> ImageSourceType;

// Registration Types
typedef itk::TranslationTransform<double, DIM> TransformType;
typedef itk::GradientDescentOptimizer OptimizerType;
typedef itk::MutualInformationImageToImageMetric
		<FixedImageType, MovingImageType> MetricType;
typedef itk::LinearInterpolateImageFunction<MovingImageType, double>
		InterpolatorType;
typedef itk::ImageRegistrationMethod<FixedImageType, MovingImageType>
		RegistrationType;
typedef RegistrationType::ParametersType ParametersType;

// Prefiltering Types
typedef itk::NormalizeImageFilter<ImageType, ImageType>
		NormalizeFilterType;
typedef itk::DiscreteGaussianImageFilter<ImageType, ImageType>
		GaussianFilterType;

// Postfiltering Types
typedef itk::ResampleImageFilter<MovingImageType, FixedImageType>
		ResampleFilterType;
typedef itk::CheckerBoardImageFilter<ImageType>
		CheckerBoardFilterType;
typedef itk::SubtractImageFilter<ImageType, ImageType, ImageType>
		DifferenceFilterType;

void printImgSize(const ReaderType::Pointer& reader) {
	// Print Size
	std::cout << "Size of fixed image is ("
			<< reader->GetOutput()->
					GetLargestPossibleRegion().GetSize()[0]
			<< ", "
			<< reader->GetOutput()->
					GetLargestPossibleRegion().GetSize()[1]
			<< ", "
			<< reader->GetOutput()->
					GetLargestPossibleRegion().GetSize()[2]
			<< ")" << std::endl;
}

ImageType::Pointer translationMulitmodalRegistration(ImageType::Pointer fixed,
		ImageType::Pointer moving) {
	// Prefiltering
	NormalizeFilterType::Pointer fixedNormalizer =
			NormalizeFilterType::New();
	NormalizeFilterType::Pointer movingNormalizer =
			NormalizeFilterType::New();
	GaussianFilterType::Pointer fixedSmoother =
			GaussianFilterType::New();
	GaussianFilterType::Pointer movingSmoother =
			GaussianFilterType::New();
	fixedNormalizer->SetInput(fixed);
	movingNormalizer->SetInput(moving);
	fixedSmoother->SetInput(fixedNormalizer->GetOutput());
	movingSmoother->SetInput(movingNormalizer->GetOutput());

	// Registration Prep
	MetricType::Pointer metric = MetricType::New();
	TransformType::Pointer transform = TransformType::New();
	OptimizerType::Pointer optimizer = OptimizerType::New();
	InterpolatorType::Pointer interpolator = InterpolatorType::New();
	RegistrationType::Pointer registration = RegistrationType::New();

	metric->SetFixedImageStandardDeviation(0.4);
	metric->SetMovingImageStandardDeviation(0.4);
	registration->SetMetric(metric);
	registration->SetOptimizer(optimizer);
	registration->SetTransform(transform);
	registration->SetInterpolator(interpolator);
	registration->SetFixedImage(fixedSmoother->GetOutput());
	registration->SetMovingImage(movingSmoother->GetOutput());

	fixedNormalizer->Update();
	FixedImageType::RegionType fixedImageRegion =
			fixedNormalizer->GetOutput()->GetBufferedRegion();
	registration->SetFixedImageRegion(fixedImageRegion);

	ParametersType initialParameters(
			transform->GetNumberOfParameters());
	registration->SetInitialTransformParameters(initialParameters);

	// final Optimisation
	const unsigned int numberOfSamples = static_cast<unsigned int>(
			fixedImageRegion.GetNumberOfPixels() * 0.01);
	optimizer->SetLearningRate(15.0);
	optimizer->SetNumberOfIterations(1000);
	optimizer->MaximizeOn();

	// Registration
	try {
		registration->Update();
	}
	catch(itk::ExceptionObject& err) {
		throw(err);
	}
	ParametersType finalParameters =
			registration->GetLastTransformParameters();
	std::cout << "Finished after "
			<< optimizer->GetCurrentIteration()
			<< " iterations" << std::endl;

	// Apply Transform
	TransformType::Pointer finalTransform = TransformType::New();
	ResampleFilterType::Pointer resample = ResampleFilterType::New();
	finalTransform->SetParameters(finalParameters);
	finalTransform->SetFixedParameters(
			transform->GetFixedParameters());
	resample->SetTransform(finalTransform);
	resample->SetInput(moving);
	resample->SetSize(fixed->GetLargestPossibleRegion().GetSize());
	resample->SetOutputOrigin(fixed->GetOrigin());
	resample->SetOutputSpacing(fixed->GetSpacing());
	resample->SetOutputDirection(fixed->GetDirection());
	resample->SetDefaultPixelValue(100);
	resample->Update();

	return resample->GetOutput();
}

int main()
{
	// Read
	ReaderType::Pointer fixedImageReader = ReaderType::New();
	ReaderType::Pointer movingImageReader = ReaderType::New();

	std::cout << "Load image (" << FIXED_FILE << ")... ";
	fixedImageReader->SetFileName(FIXED_FILE);
	std::cout << "Success"<< std::endl;
	std::cout << "Load image (" << MOVING_FILE << ")... ";
	movingImageReader->SetFileName(MOVING_FILE);
	std::cout << "Success"<< std::endl;
	fixedImageReader->Update();
	movingImageReader->Update();

	// Print Size
	printImgSize(fixedImageReader);
	printImgSize(movingImageReader);

	ImageType::Pointer registered;
	try {
		registered = translationMulitmodalRegistration(
				fixedImageReader->GetOutput(),
				movingImageReader->GetOutput());
	}
	catch(itk::ExceptionObject& err) {
		std::cerr << "ExceptionObject caught!" << std::endl;
		std::cerr << err << std::endl;
		return EXIT_FAILURE;
	}

	// Stitch images together
	CheckerBoardFilterType::Pointer checker =
			CheckerBoardFilterType::New();
	DifferenceFilterType::Pointer difference =
			DifferenceFilterType::New();
	checker->SetInput1(fixedImageReader->GetOutput());
	checker->SetInput2(registered);
	difference->SetInput1(fixedImageReader->GetOutput());
	difference->SetInput2(registered);

	// Write Out
	std::cout << "Writing file to " << OUT_FILE << "... ";
	WriterType::Pointer outWriter = WriterType::New();
	WriterType::Pointer difWriter = WriterType::New();
	outWriter->SetFileName(OUT_FILE);
	outWriter->SetInput(checker->GetOutput());
	outWriter->Update();
	difWriter->SetFileName(DIFF_FILE);
	difWriter->SetInput(difference->GetOutput());
	difWriter->Update();
	std::cout << "Success"<< std::endl;

	return 0;
}
