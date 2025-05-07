import React, { useRef } from 'react';

// This is for the drag and drop ---------------------------------------------

type Props = {
  onFileSelect: (file: File) => void;
};

const ImageDropZone: React.FC<Props> = ({ onFileSelect }) => {
  const inputRef = useRef<HTMLInputElement | null>(null);

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) onFileSelect(file);
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const handleClick = () => {
    inputRef.current?.click();
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onFileSelect(file);
  };

  return (
    <>
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onClick={handleClick}
        className="w-full max-w-md h-48 border-4 border-dashed border-gray-400 rounded-lg flex items-center justify-center text-gray-500 cursor-pointer mb-4"
      >
        <p>Drag and drop an image here, or click to browse</p>
      </div>

      <input
        type="file"
        accept="image/*"
        ref={inputRef}
        onChange={handleInputChange}
        className="hidden"
      />
    </>
  );
};

export default ImageDropZone;
