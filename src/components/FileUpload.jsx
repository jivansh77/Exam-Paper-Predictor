import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { CloudArrowUpIcon } from '@heroicons/react/24/outline';

export default function FileUpload({ onFileUpload }) {
  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) onFileUpload(file);
  }, [onFileUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    multiple: false
  });

  return (
    <div className="max-w-xl mx-auto mt-8">
      <div
        {...getRootProps()}
        className={`p-10 border-2 border-dashed rounded-lg text-center transition-all duration-300 cursor-pointer
          ${isDragActive ? 'border-blue-400 bg-blue-50' : 'border-gray-300 hover:border-blue-300 hover:bg-gray-100'}`}
      >
        <input {...getInputProps()} />
        <CloudArrowUpIcon className="mx-auto h-12 w-12 text-gray-400" />
        <p className="mt-2 text-gray-600">
          Drop PDF file here or click to select
        </p>
        <p className="mt-1 text-sm text-gray-500">
          (Only .pdf files are accepted)
        </p>
      </div>
    </div>
  );
}