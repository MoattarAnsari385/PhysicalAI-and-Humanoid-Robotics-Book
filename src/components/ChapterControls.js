import React from 'react';
import PersonalizeToggle from './PersonalizeToggle';
import LanguageToggle from './LanguageToggle';

const ChapterControls = () => {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      gap: '1rem',
      padding: '1rem',
      marginBottom: '2rem',
      borderRadius: '8px',
      background: 'linear-gradient(135deg, rgba(224, 0, 112, 0.05), rgba(255, 122, 0, 0.05))',
      border: '1px solid rgba(224, 0, 112, 0.2)',
      fontFamily: 'system-ui, -apple-system, sans-serif'
    }}>
      <h3 style={{
        margin: '0 0 0.5rem 0',
        color: '#E00070',
        fontSize: '1.1rem',
        fontWeight: '600'
      }}>
        Customize Your Learning Experience
      </h3>
      <div style={{
        display: 'flex',
        gap: '1rem',
        flexWrap: 'wrap'
      }}>
        <PersonalizeToggle />
        <LanguageToggle />
      </div>
    </div>
  );
};

export default ChapterControls;