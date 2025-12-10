import React, { useEffect } from 'react';
import { useUser } from '../contexts/UserContext';

const LanguageToggle = () => {
  const { preferences, setPreferences } = useUser();

  const handleLanguageChange = (language) => {
    setPreferences({ language });
    // Apply RTL layout if needed
    applyRTL(language === 'ur');
  };

  const applyRTL = (isRTL) => {
    document.documentElement.dir = isRTL ? 'rtl' : 'ltr';
    document.documentElement.lang = isRTL ? 'ur' : 'en';

    // Apply RTL-specific styles
    if (isRTL) {
      document.documentElement.style.setProperty('--text-direction', 'rtl');
      document.documentElement.style.setProperty('--text-align', 'right');
    } else {
      document.documentElement.style.setProperty('--text-direction', 'ltr');
      document.documentElement.style.setProperty('--text-align', 'left');
    }
  };

  // Apply initial RTL setting when component mounts
  useEffect(() => {
    applyRTL(preferences.language === 'ur');
  }, []);

  return (
    <div className="language-toggle" style={{
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
        🌐 Language:
      </span>
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
            color: preferences.language === lang ? 'white' : '#333',
            cursor: 'pointer',
            fontSize: '0.8rem',
            fontWeight: '500',
            transition: 'all 0.3s ease',
            textTransform: 'uppercase'
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