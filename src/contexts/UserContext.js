import React, { createContext, useContext, useReducer, useEffect } from 'react';
import { useAuth } from '@better-auth/react';
import { mockApi } from '../utils/api';

const UserContext = createContext();

const initialState = {
  user: null,
  preferences: {
    level: 'intermediate',
    language: 'en'
  },
  isAuthenticated: false,
  loading: true
};

function userReducer(state, action) {
  switch (action.type) {
    case 'SET_USER':
      return {
        ...state,
        user: action.payload,
        isAuthenticated: !!action.payload,
        loading: false
      };
    case 'SET_PREFERENCES':
      return {
        ...state,
        preferences: { ...state.preferences, ...action.payload }
      };
    case 'SET_LOADING':
      return {
        ...state,
        loading: action.payload
      };
    case 'LOGOUT':
      return {
        ...initialState,
        loading: false
      };
    default:
      return state;
  }
}

export const UserProvider = ({ children }) => {
  const [state, dispatch] = useReducer(userReducer, initialState);
  const { session, isPending } = useAuth();

  useEffect(() => {
    if (!isPending) {
      if (session?.user) {
        dispatch({
          type: 'SET_USER',
          payload: {
            ...session.user,
            id: session.user.id,
            email: session.user.email,
            name: session.user.name,
            softwareLevel: session.user.softwareLevel,
            gpuType: session.user.gpuType,
            jetsonModel: session.user.jetsonModel
          }
        });

        // Load user preferences from database or localStorage
        loadUserPreferences(session.user.id);
      } else {
        dispatch({ type: 'SET_USER', payload: null });
      }
    }
  }, [session, isPending]);

  const loadUserPreferences = async (userId) => {
    try {
      const data = await mockApi.get(`/api/user-preferences/${userId}`);
      dispatch({
        type: 'SET_PREFERENCES',
        payload: {
          level: data.level_preference || 'intermediate',
          language: data.language || 'en'
        }
      });
    } catch (error) {
      console.error('Error loading user preferences:', error);
    }
  };

  const setUser = (user) => {
    dispatch({ type: 'SET_USER', payload: user });
  };

  const setPreferences = (preferences) => {
    dispatch({ type: 'SET_PREFERENCES', payload: preferences });

    // Update in database
    if (state.user?.id) {
      updatePreferenceInDB(state.user.id, preferences);
    }
  };

  const updatePreferenceInDB = async (userId, preferences) => {
    try {
      await mockApi.put('/api/user-preferences', {
        user_id: userId,
        ...preferences
      });
    } catch (error) {
      console.error('Error updating preferences:', error);
    }
  };

  const logout = () => {
    dispatch({ type: 'LOGOUT' });
  };

  return (
    <UserContext.Provider value={{ ...state, setUser, setPreferences, logout }}>
      {children}
    </UserContext.Provider>
  );
};

export const useUser = () => {
  const context = useContext(UserContext);
  if (!context) {
    throw new Error('useUser must be used within a UserProvider');
  }
  return context;
};