import React from 'react';

// This is a custom App wrapper - AuthContext is provided in Root.js
const AppWrapper = ({ children }) => {
  return (
    <>
      {children}
    </>
  );
};

export default AppWrapper;