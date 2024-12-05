import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

export const analyzeService = {
  async uploadPaper(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await axios.post(`${API_URL}/analyze`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return response.data;
  },

  async getTopics() {
    const response = await axios.get(`${API_URL}/topics`);
    return response.data;
  },

  async getQuestions(topicId) {
    const response = await axios.get(`${API_URL}/questions/${topicId}`);
    return response.data;
  }
};

export default analyzeService;