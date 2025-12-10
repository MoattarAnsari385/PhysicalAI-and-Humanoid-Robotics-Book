import React from 'react';
import { useUser } from '../contexts/UserContext';

const PersonalizeToggle = () => {
  const { preferences, setPreferences } = useUser();

  const handleLevelChange = (level) => {
    setPreferences({ level });
  };

  return (
    <div className="personalize-toggle" style={{
      display: 'flex',
      alignItems: 'center',
      gap: '0.5rem',
      padding: '0.5rem 1rem',
      borderRadius: '8px',
      background: 'rgba(224, 0, 112, 0.1)',
      border: '1px solid rgba(224, 0, 112, 0.3)',
      fontFamily: 'system-ui, -apple-system, sans-serif',
      fontSize: '0.9rem'
    }}>
      <span style={{
        color: '#E00070',
        fontWeight: '500',
        display: 'flex',
        alignItems: 'center',
        gap: '0.25rem'
      }}>
        ⚙️ Personalize:
      </span>
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
            color: preferences.level === level ? 'white' : '#333',
            cursor: 'pointer',
            fontSize: '0.8rem',
            fontWeight: '500',
            transition: 'all 0.3s ease',
            textTransform: 'capitalize'
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