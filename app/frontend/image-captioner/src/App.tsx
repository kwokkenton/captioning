import { useState, useCallback } from "react";
import ImageDropZone from "./ImageDropZone";
import WebcamCapture from "./WebcamCapture";
import SpeechSynthesis from "./SpeechSynthesis";

function App() {
  // This sets the application state and the functions to change them
  const [caption, setCaption] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isWebcamMode, setIsWebcamMode] = useState<boolean>(false); // State to toggle between modes
  const [attentionImages, setAttentionImages] = useState<
    { token: string; attention_image: string }[]
  >([]);
  // This is for the drag and drop ---------------------------------------------

  // Runs when a file is selected/ dropped
  const handleFile = useCallback((file: File | null) => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);
    // Generates temporary local URL
    setPreviewUrl(URL.createObjectURL(file));
    setCaption("");
    setLoading(true);

    fetch("http://localhost:8000/caption", {
      method: "POST",
      body: formData,
    })
      .then((res) => res.json())
      .then((data: { caption: string }) => {
        setCaption(data.caption);
        setAttentionImages(data.attention_images);
      })
      .catch((err) => {
        console.error(err);
        setCaption("Error generating caption.");
      })
      .finally(() => {
        setLoading(false);
      });
  }, []);
  // -----------------------------------------------------------------------------

  // This is for the drag and drop ---------------------------------------------

  // -----------------------------------------------------------------------------
  return (
    <div className="min-h-screen min-w-screen bg-black-900 flex flex-col items-center justify-center p-4">
      <h1 className="text-2xl font-bold mb-4 bg-gradient-to-r from-green-600 to-blue-300 text-transparent bg-clip-text">
        Image Captioner
      </h1>
      <div className="mb-4">
        {/* Toggle between webcam and file upload */}
        <button
          onClick={() => setIsWebcamMode(!isWebcamMode)}
          className="px-4 py-2 bg-gray-600 text-white rounded shadow hover:bg-gray-700"
        >
          {isWebcamMode ? "Switch to File Upload" : "Switch to Webcam"}
        </button>
      </div>

      {/* Conditional rendering based on `isWebcamMode` */}
      {isWebcamMode ? (
        <WebcamCapture onCapture={handleFile} />
      ) : (
        <ImageDropZone onFileSelect={handleFile} />
      )}

      {previewUrl && (
        <div className="mt-4">
          <img
            src={previewUrl}
            alt="Preview"
            className="max-w-md max-h-80 rounded shadow"
          />
        </div>
      )}

      {loading && <p className="text-blue-500 mt-4">Generating caption...</p>}

      {caption && !loading && (
        <div className="mt-4 p-4 bg-white rounded shadow max-w-md text-center">
          <p className="text-gray-700 font-semibold">Caption:</p>
          <p className="text-gray-900">{caption}</p>
        </div>
      )}

      {/* Horizontal Scrolling of Attention Images */}
      <div>
        {attentionImages.length > 0 && !loading && (
          <div className="mt-4 w-[calc(100%-2rem)] overflow-x-auto flex space-x-4 px-4 py-2 max-w-full h-[300px]">
            {/* Set a fixed height for the container */}
            {attentionImages.map((item, index) => (
              <div key={index} className="flex-shrink-0 w-56 text-center">
                <h3 className="text-lg font-semibold text-white">
                  {item.token}
                </h3>
                <img
                  src={`data:image/png;base64,${item.attention_image}`}
                  alt={`Attention map for token: ${item.token}`}
                  className="mx-auto rounded-lg shadow-lg"
                  style={{ width: "224px", height: "224px" }} // Adjust based on your attention map size
                />
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Add the SpeechSynthesis component here */}
      <SpeechSynthesis text={caption} />
    </div>
  );
}

export default App;
