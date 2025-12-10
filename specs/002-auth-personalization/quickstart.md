# Quickstart Guide: Authentication + Personalization + Urdu Translation

**Feature**: `002-auth-personalization`
**Created**: 2025-12-10
**Status**: Complete

## Overview

This guide provides a quick start for implementing the authentication, personalization, and Urdu translation features in the Physical AI & Humanoid Robotics Book.

## Prerequisites

- Node.js 18+ installed
- Docusaurus project set up
- Better-Auth account or self-hosted instance
- Database (Supabase or Neon Postgres) credentials
- Basic knowledge of React and Docusaurus

## Setup Steps

### 1. Install Dependencies

```bash
npm install @better-auth/react @better-auth/node @docusaurus/core react react-dom
```

### 2. Database Setup

Run these SQL commands to create the required tables:

```sql
-- Create users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    software_level VARCHAR(20) CHECK (software_level IN ('beginner', 'intermediate', 'advanced')),
    gpu_type VARCHAR(100),
    jetson_model VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create user_preferences table
CREATE TABLE user_preferences (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    level_preference VARCHAR(20) CHECK (level_preference IN ('beginner', 'intermediate', 'advanced')),
    language VARCHAR(5) CHECK (language IN ('en', 'ur')) DEFAULT 'en',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_user_preferences_user_id ON user_preferences(user_id);
```

### 3. Better-Auth Configuration

Create `src/auth/better-auth.js`:

```javascript
import { betterAuth } from "@better-auth/node";
import { postgresAdapter } from "@better-auth/postgres-adapter";
import { Pool } from 'pg';

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});

export const auth = betterAuth({
  database: postgresAdapter(pool, {
    provider: "postgres",
  }),
  emailAndPassword: {
    enabled: true,
    requireEmailVerification: false,
  },
  user: {
    additionalFields: {
      softwareLevel: {
        type: "string",
        required: false,
      },
      gpuType: {
        type: "string",
        required: false,
      },
      jetsonModel: {
        type: "string",
        required: false,
      },
    },
  },
});
```

### 4. Create User Context

Create `src/contexts/UserContext.js`:

```javascript
import React, { createContext, useContext, useReducer } from 'react';

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

  const setUser = (user) => {
    dispatch({ type: 'SET_USER', payload: user });
  };

  const setPreferences = (preferences) => {
    dispatch({ type: 'SET_PREFERENCES', payload: preferences });
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
```

### 5. Create Toggle Components

Create `src/components/PersonalizeToggle.js`:

```javascript
import React from 'react';
import { useUser } from '../contexts/UserContext';

const PersonalizeToggle = () => {
  const { preferences, setPreferences } = useUser();

  const handleLevelChange = (level) => {
    setPreferences({ level });
    // Update in database
    updatePreferenceInDB('level_preference', level);
  };

  const updatePreferenceInDB = async (field, value) => {
    // Implementation to update user preferences in database
    try {
      const response = await fetch('/api/user-preferences', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ [field]: value }),
      });
      if (!response.ok) {
        throw new Error('Failed to update preferences');
      }
    } catch (error) {
      console.error('Error updating preferences:', error);
    }
  };

  return (
    <div className="personalize-toggle" style={{
      display: 'flex',
      alignItems: 'center',
      gap: '0.5rem',
      padding: '0.5rem 1rem',
      borderRadius: '8px',
      background: 'rgba(224, 0, 112, 0.1)',
      border: '1px solid rgba(224, 0, 112, 0.3)'
    }}>
      <span style={{ fontSize: '0.9rem', color: '#E00070' }}>⚙️ Personalize:</span>
      {['beginner', 'intermediate', 'advanced'].map((level) => (
        <button
          key={level}
          onClick={() => handleLevelChange(level)}
          style={{
            padding: '0.25rem 0.75rem',
            borderRadius: '4px',
            border: 'none',
            background: preferences.level === level
              ? 'linear-gradient(45deg, #E00070, #FF7A00)'
              : 'rgba(255, 255, 255, 0.1)',
            color: preferences.level === level ? 'white' : '#ffffff',
            cursor: 'pointer',
            fontSize: '0.8rem',
            transition: 'all 0.3s ease'
          }}
          onMouseEnter={(e) => {
            if (preferences.level !== level) {
              e.target.style.background = 'rgba(224, 0, 112, 0.2)';
            }
          }}
          onMouseLeave={(e) => {
            if (preferences.level !== level) {
              e.target.style.background = 'rgba(255, 255, 255, 0.1)';
            }
          }}
        >
          {level.charAt(0).toUpperCase() + level.slice(1)}
        </button>
      ))}
    </div>
  );
};

export default PersonalizeToggle;
```

Create `src/components/LanguageToggle.js`:

```javascript
import React from 'react';
import { useUser } from '../contexts/UserContext';

const LanguageToggle = () => {
  const { preferences, setPreferences } = useUser();

  const handleLanguageChange = (language) => {
    setPreferences({ language });
    // Update in database and apply RTL if needed
    updatePreferenceInDB('language', language);
    applyRTL(language === 'ur');
  };

  const updatePreferenceInDB = async (field, value) => {
    // Implementation to update user preferences in database
    try {
      const response = await fetch('/api/user-preferences', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ [field]: value }),
      });
      if (!response.ok) {
        throw new Error('Failed to update preferences');
      }
    } catch (error) {
      console.error('Error updating preferences:', error);
    }
  };

  const applyRTL = (isRTL) => {
    document.documentElement.dir = isRTL ? 'rtl' : 'ltr';
    document.documentElement.lang = isRTL ? 'ur' : 'en';
  };

  return (
    <div className="language-toggle" style={{
      display: 'flex',
      alignItems: 'center',
      gap: '0.5rem',
      padding: '0.5rem 1rem',
      borderRadius: '8px',
      background: 'rgba(224, 0, 112, 0.1)',
      border: '1px solid rgba(224, 0, 112, 0.3)'
    }}>
      <span style={{ fontSize: '0.9rem', color: '#E00070' }}>🌐 Language:</span>
      {['en', 'ur'].map((lang) => (
        <button
          key={lang}
          onClick={() => handleLanguageChange(lang)}
          style={{
            padding: '0.25rem 0.75rem',
            borderRadius: '4px',
            border: 'none',
            background: preferences.language === lang
              ? 'linear-gradient(45deg, #E00070, #FF7A00)'
              : 'rgba(255, 255, 255, 0.1)',
            color: preferences.language === lang ? 'white' : '#ffffff',
            cursor: 'pointer',
            fontSize: '0.8rem',
            transition: 'all 0.3s ease'
          }}
          onMouseEnter={(e) => {
            if (preferences.language !== lang) {
              e.target.style.background = 'rgba(224, 0, 112, 0.2)';
            }
          }}
          onMouseLeave={(e) => {
            if (preferences.language !== lang) {
              e.target.style.background = 'rgba(255, 255, 255, 0.1)';
            }
          }}
        >
          {lang.toUpperCase()}
        </button>
      ))}
    </div>
  );
};

export default LanguageToggle;
```

### 6. Protect Routes

Create a protected route component:

```javascript
// src/components/ProtectedRoute.js
import React, { useEffect } from 'react';
import { useUser } from '../contexts/UserContext';
import { useNavigate } from 'react-router-dom';

const ProtectedRoute = ({ children }) => {
  const { isAuthenticated, loading } = useUser();
  const navigate = useNavigate();

  useEffect(() => {
    if (!loading && !isAuthenticated) {
      navigate('/login');
    }
  }, [isAuthenticated, loading, navigate]);

  if (loading) {
    return <div>Loading...</div>;
  }

  return isAuthenticated ? children : null;
};

export default ProtectedRoute;
```

## Environment Variables

Add these to your `.env` file:

```env
DATABASE_URL=your_database_connection_string
BETTER_AUTH_URL=your_auth_url
BETTER_AUTH_SECRET=your_secret_key
```

## Integration with Docusaurus

1. Wrap your application with UserProvider in the main layout
2. Add authentication pages to the Docusaurus routing
3. Integrate toggle components into chapter layouts
4. Update existing components to use the UserContext

## Testing

### Unit Tests
- Test authentication flow
- Test preference saving/loading
- Test context provider functionality
- Test toggle component behavior

### Integration Tests
- End-to-end user journey
- Database integration
- Preference persistence
- RTL layout switching

## Next Steps

1. Implement the API endpoints for user preferences
2. Create the signup and login pages with background questions
3. Integrate the components into the book reader
4. Test the complete user flow
5. Add proper error handling and validation