import React from 'react';

// This is the main App wrapper - AuthContext is provided in Root.js
function App({children}) {
  return (
    <>
      {children}
    </>
  );
}

export default App;