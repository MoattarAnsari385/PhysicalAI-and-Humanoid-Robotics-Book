import React from 'react';
import { useUser } from '../contexts/UserContext';

const PersonalizedContent = ({ content }) => {
  const { preferences } = useUser();

  // Function to get content based on user level
  const getContentForLevel = (content, level) => {
    if (!content || typeof content !== 'object') {
      return content || '';
    }

    // Return content specific to user level, or default content
    return content[level] || content.default || content.intermediate || '';
  };

  // Function to get translated content
  const getTranslatedContent = (content, language) => {
    if (!content || typeof content !== 'object') {
      return content || '';
    }

    // If content has translations, return the appropriate one
    if (content.translations && content.translations[language]) {
      return content.translations[language];
    }

    // Otherwise return the original content
    return content.default || content;
  };

  // Get content based on user preferences
  const levelSpecificContent = getContentForLevel(content, preferences.level);
  const finalContent = getTranslatedContent(levelSpecificContent, preferences.language);

  // If final content is an object with HTML, render it safely
  if (typeof finalContent === 'object' && finalContent.html) {
    return (
      <div
        className="personalized-content"
        dangerouslySetInnerHTML={{ __html: finalContent.html }}
      />
    );
  }

  // Otherwise render as text
  return (
    <div className="personalized-content">
      {finalContent}
    </div>
  );
};

export default PersonalizedContent;