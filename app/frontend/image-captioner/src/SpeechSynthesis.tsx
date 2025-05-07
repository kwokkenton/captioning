import React, { useEffect } from 'react';

type Props = {
  text: string; // Text to be read out loud
};

const SpeechSynthesis: React.FC<Props> = ({ text }) => {
  useEffect(() => {
    if (text) {
      const speech = new SpeechSynthesisUtterance(text);
      speech.lang = 'en-US'; // You can change the language as needed

      window.speechSynthesis.speak(speech);
    }
  }, [text]); // Runs when `text` changes

  return null; // This component doesn't need to render anything
};

export default SpeechSynthesis;
