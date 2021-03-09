package com.yay.vehicle;

import com.google.cloud.vision.v1.*;
import com.google.cloud.vision.v1.Feature.Type;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.cloud.gcp.vision.CloudVisionTemplate;
import org.springframework.core.io.ResourceLoader;
import org.springframework.ui.ModelMap;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.ModelAndView;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Code sample demonstrating Cloud Vision usage within the context of Spring Framework using Spring
 * Cloud GCP libraries. The sample is written as a Spring Boot application to demonstrate a
 * practical application of this usage.
 */
@RestController
public class VisionController {

    @Autowired private ResourceLoader resourceLoader;

    // [START spring_vision_autowire]
    @Autowired private CloudVisionTemplate cloudVisionTemplate;
    // [END spring_vision_autowire]

    /**
     * This method downloads an image from a URL and sends its contents to the Vision API for label
     * detection.
     *
     * @param imageUrl the URL of the image
     * @param map the model map to use
     * @return a string with the list of labels and percentage of certainty
     */
    @GetMapping("/extractLabels")
    public ModelAndView extractLabels(String imageUrl, ModelMap map) {
        // [START spring_vision_image_labelling]
        AnnotateImageResponse response =
                this.cloudVisionTemplate.analyzeImage(
                        this.resourceLoader.getResource(imageUrl), Type.LABEL_DETECTION);

        Map<String, Float> imageLabels =
                response
                        .getLabelAnnotationsList()
                        .stream()
                        .collect(
                                Collectors.toMap(
                                        EntityAnnotation::getDescription,
                                        EntityAnnotation::getScore,
                                        (u, v) -> {
                                            throw new IllegalStateException(String.format("Duplicate key %s", u));
                                        },
                                        LinkedHashMap::new));
        // [END spring_vision_image_labelling]

        map.addAttribute("annotations", imageLabels);
        map.addAttribute("imageUrl", imageUrl);

        return new ModelAndView("result", map);
    }

    @GetMapping("/extractText")
    public String extractText(String imageUrl) {
        // [START spring_vision_text_extraction]
        String textFromImage =
                this.cloudVisionTemplate.extractTextFromImage(this.resourceLoader.getResource(imageUrl));

        /*try {
            detectLocalizedObjectsGcs(imageUrl);
        } catch (IOException e) {
            System.out.println("IOException: " + e.getMessage());
        }*/
        return "Text from image: " + textFromImage;
        // [END spring_vision_text_extraction]
    }

    /**
     * Detects localized objects in a remote image on Google Cloud Storage.
     *
     * @param gcsPath The path to the remote file on Google Cloud Storage to detect localized objects
     *     on.
     * @throws Exception on errors while closing the client.
     * @throws IOException on Input/Output errors.
     */
    public void detectLocalizedObjectsGcs(String gcsPath) throws IOException {
        List<AnnotateImageRequest> requests = new ArrayList<>();

        ImageSource imgSource = ImageSource.newBuilder().setGcsImageUri(gcsPath).build();
        Image img = Image.newBuilder().setSource(imgSource).build();

        AnnotateImageRequest request =
                AnnotateImageRequest.newBuilder()
                        .addFeatures(Feature.newBuilder().setType(Type.OBJECT_LOCALIZATION))
                        .setImage(img)
                        .build();
        requests.add(request);

        // Initialize client that will be used to send requests. This client only needs to be created
        // once, and can be reused for multiple requests. After completing all of your requests, call
        // the "close" method on the client to safely clean up any remaining background resources.
        try (ImageAnnotatorClient client = ImageAnnotatorClient.create()) {
            // Perform the request
            BatchAnnotateImagesResponse response = client.batchAnnotateImages(requests);
            List<AnnotateImageResponse> responses = response.getResponsesList();
            client.close();
            // Display the results
            for (AnnotateImageResponse res : responses) {
                for (LocalizedObjectAnnotation entity : res.getLocalizedObjectAnnotationsList()) {
                    System.out.format("Object name: %s%n", entity.getName());
                    System.out.format("Confidence: %s%n", entity.getScore());
                    System.out.format("Normalized Vertices:%n");
                    entity
                            .getBoundingPoly()
                            .getNormalizedVerticesList()
                            .forEach(vertex -> System.out.format("- (%s, %s)%n", vertex.getX(), vertex.getY()));
                }
            }
        }
    }

}
