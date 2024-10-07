### From https://tiwarinitin1999.medium.com/yolov10-to-litert-object-detection-on-android-with-google-ai-edge-2d0de5619e71

!pip install ultralytics
!pip install ai-edge-model-explorer
!pip install ai-edge-litert



model = YOLO("yolov10n.pt")
model.export(format="tflite")

LITE_RT_EXPORT_PATH = "yolov10n_saved_model/" # @param {type : 'string'}
LITE_RT_MODEL = "yolov10n_float16.tflite" # @param {type : 'string'}
LITE_RT_MODEL_PATH = LITE_RT_EXPORT_PATH + LITE_RT_MODEL

model_explorer.visualize(LITE_RT_MODEL_PATH)

interpreter = Interpreter(model_path = LITE_RT_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Model input size: {input_size}")
print(f"Output tensor shape: {output_details[0]['shape']}")

def detect(input_data, is_video_frame=False):
    input_size = input_details[0]['shape'][1]

    if is_video_frame:
        original_height, original_width = input_data.shape[:2]
        image = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (input_size, input_size))
        image = image / 255.0
    else:
        image, (original_height, original_width) = load_image(input_data, input_size)

    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(image, axis=0).astype(np.float32))
    interpreter.invoke()

    output_data = [interpreter.get_tensor(detail['index']) for detail in output_details]
    return output_data, (original_height, original_width)



# Postprocess the output.
def postprocess_output(output_data, original_dims, labels, confidence_threshold):
  output_tensor = output_data[0]
  detections = []
  original_height, original_width = original_dims

  for i in range(output_tensor.shape[1]):
    box = output_tensor[0, i, :4]
    confidence = output_tensor[0, i, 4]
    class_id = int(output_tensor[0, i, 5])

    if confidence > confidence_threshold:
      x_min = int(box[0] * original_width)
      y_min = int(box[1] * original_height)
      x_max = int(box[2] * original_width)
      y_max = int(box[3] * original_height)

      label_name = labels.get(str(class_id), "Unknown")

      detections.append({
          "box": [y_min, x_min, y_max, x_max],
          "score": confidence,
          "class": class_id,
          "label": label_name
      })

  return detections


###### Deploy the model on Android
const val MODEL_PATH = "yolov10n_float16.tflite" 

#Detector.kt
// Detects the objects.
class Detector(
    private val context: Context,
    private val modelPath: String,
    private val labelPath: String?,
    private val detectorListener: DetectorListener,
    private val message: (String) -> Unit
) {
    private var interpreter: Interpreter
    private var labels = mutableListOf<String>()

    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0

    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
        .add(CastOp(INPUT_IMAGE_TYPE))
        .build()

    init {
        val options = Interpreter.Options().apply{
            this.setNumThreads(4)
        }

        val model = FileUtil.loadMappedFile(context, modelPath)
        interpreter = Interpreter(model, options)

        labels.addAll(extractNamesFromMetadata(model))
        if (labels.isEmpty()) {
            if (labelPath == null) {
                message("Model not contains metadata, provide LABELS_PATH in Constants.kt")
                labels.addAll(MetaData.TEMP_CLASSES)
            } else {
                labels.addAll(extractNamesFromLabelFile(context, labelPath))
            }
        }

        labels.forEach(::println)

        val inputShape = interpreter.getInputTensor(0)?.shape()
        val outputShape = interpreter.getOutputTensor(0)?.shape()

        if (inputShape != null) {
            tensorWidth = inputShape[1]
            tensorHeight = inputShape[2]

            // If in case input shape is in format of [1, 3, ..., ...]
            if (inputShape[1] == 3) {
                tensorWidth = inputShape[2]
                tensorHeight = inputShape[3]
            }
        }

        if (outputShape != null) {
            numElements = outputShape[1]
            numChannel = outputShape[2]
        }
    }

// Extracts bounding box, label, confidence.
private fun bestBox(array: FloatArray) : List<BoundingBox> {
    val boundingBoxes = mutableListOf<BoundingBox>()
    for (r in 0 until numElements) {
        val cnf = array[r * numChannel + 4]
        if (cnf > CONFIDENCE_THRESHOLD) {
            val x1 = array[r * numChannel]
            val y1 = array[r * numChannel + 1]
            val x2 = array[r * numChannel + 2]
            val y2 = array[r * numChannel + 3]
            val cls = array[r * numChannel + 5].toInt()
            val clsName = labels[cls]
            boundingBoxes.add(
                BoundingBox(
                    x1 = x1, y1 = y1, x2 = x2, y2 = y2,
                    cnf = cnf, cls = cls, clsName = clsName
                )
            )
        }
    }
    return boundingBoxes
}

# OverlayView.kt
class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results = listOf<BoundingBox>()
    private val boxPaint = Paint()
    private val textBackgroundPaint = Paint()
    private val textPaint = Paint()

    private var bounds = Rect()
    private val colorMap = mutableMapOf<String, Int>()

    init {
        initPaints()
    }

    fun clear() {
        results = listOf()
        textPaint.reset()
        textBackgroundPaint.reset()
        boxPaint.reset()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        textBackgroundPaint.color = Color.WHITE
        textBackgroundPaint.style = Paint.Style.FILL
        textBackgroundPaint.textSize = 42f

        textPaint.color = Color.WHITE
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 42f
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        results.forEach { boundingBox ->
            // Get or create a color for this label
            val color = getColorForLabel(boundingBox.clsName)
            boxPaint.color = color
            boxPaint.strokeWidth = 8F
            boxPaint.style = Paint.Style.STROKE

            val left = boundingBox.x1 * width
            val top = boundingBox.y1 * height
            val right = boundingBox.x2 * width
            val bottom = boundingBox.y2 * height

            canvas.drawRoundRect(left, top, right, bottom, 16f, 16f, boxPaint)

            val drawableText = "${boundingBox.clsName} ${Math.round(boundingBox.cnf * 100.0) / 100.0}"

            textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
            val textWidth = bounds.width()
            val textHeight = bounds.height()

            val textBackgroundRect = RectF(
                left,
                top,
                left + textWidth + BOUNDING_RECT_TEXT_PADDING,
                top + textHeight + BOUNDING_RECT_TEXT_PADDING
            )
            textBackgroundPaint.color = color // Set background color same as bounding box
            canvas.drawRoundRect(textBackgroundRect, 8f, 8f, textBackgroundPaint)

            canvas.drawText(drawableText, left, top + textHeight, textPaint)
        }
    }

    private fun getColorForLabel(label: String): Int {
        return colorMap.getOrPut(label) {
            // Generate a random color or you can use a predefined set of colors
            Color.rgb((0..255).random(), (0..255).random(), (0..255).random())
        }
    }

    fun setResults(boundingBoxes: List<BoundingBox>) {
        results = boundingBoxes
        invalidate()
    }

    companion object {
        private const val BOUNDING_RECT_TEXT_PADDING = 8
    }
}
