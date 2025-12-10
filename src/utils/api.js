// src/utils/api.js
// API utility functions for user preferences

export const getUserPreferences = async (userId) => {
  try {
    // In a real implementation, this would call your backend API
    // For now, we'll simulate with localStorage as fallback
    const stored = localStorage.getItem(`user-preferences-${userId}`);
    if (stored) {
      return JSON.parse(stored);
    }

    // Default preferences
    return {
      level_preference: 'intermediate',
      language: 'en',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    };
  } catch (error) {
    console.error('Error getting user preferences:', error);
    return {
      level_preference: 'intermediate',
      language: 'en',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    };
  }
};

export const updateUserPreferences = async (userId, preferences) => {
  try {
    // In a real implementation, this would call your backend API
    // For now, we'll store in localStorage
    const existing = await getUserPreferences(userId);
    const updatedPreferences = {
      ...existing,
      ...preferences,
      updated_at: new Date().toISOString()
    };

    localStorage.setItem(`user-preferences-${userId}`, JSON.stringify(updatedPreferences));
    return updatedPreferences;
  } catch (error) {
    console.error('Error updating user preferences:', error);
    throw error;
  }
};

// Mock API endpoints for development
export const mockApi = {
  get: async (endpoint) => {
    if (endpoint.startsWith('/api/user-preferences/')) {
      const userId = endpoint.split('/').pop();
      return await getUserPreferences(userId);
    }
    throw new Error(`Unknown endpoint: ${endpoint}`);
  },

  put: async (endpoint, data) => {
    if (endpoint === '/api/user-preferences') {
      const { user_id, ...preferences } = data;
      return await updateUserPreferences(user_id, preferences);
    }
    throw new Error(`Unknown endpoint: ${endpoint}`);
  }
};