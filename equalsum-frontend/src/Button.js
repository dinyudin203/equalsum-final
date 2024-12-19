import React from 'react';

const Button = ({ onClick, children, style }) => {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '5px 10px',
        borderRadius: '5px',
        cursor: 'pointer',
        ...style, // 외부 스타일 적용
      }}
    >
      {children}
    </button>
  );
};

export default Button;
