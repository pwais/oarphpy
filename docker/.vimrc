" Indenting options
set paste        " Allow pasting from OS X
set autoindent   " Keep the indent level when hitting Return
set smartindent  " Use smart indenting (mostly useful for C/C++ files)
set cindent      " Don't indent Python really poorly
set tabstop=4    " Make tabs appear four spaces wide (default is 8 spaces)
set shiftwidth=4
"set noexpandtab  " Use hard tabs please! Watch out for files with soft tabs
                 " that don't have a modeline present, especially Python files.
set nocompatible " Don't care about VI-compatibility
set fo=tcoqan    " Options for formatting text (i.e. for use with gq)

" UI stuff
set showmatch   " Show matching parens as they come up
set ruler       " Show the column number in the status bar
set rulerformat=%30(%=\:b%n%y%m%r%w\ %l,%c%V\ %P%)
set incsearch   " Find as you type
set hlsearch
set wildmenu    " Get a cool menu for tab completing file names
set lz          " Don't redraw the screen in the middle of executing macros
set visualbell t_vb=
set smartcase   " Ignore case, unless caps are used in the search
behave xterm    " Just in case...
set lbr         " Wrap only at word boundaries (default is at any character)
set cursorline
set showcmd 	" show partial commands in status line and
				" selected characters/lines in visual mode

" Syntax Highlighting
syntax enable   " Who wouldn't want syntax highlighting?
colorscheme elflord
let python_highlight_all = 1
let python_highlight_indent_errors = 0
let python_highlight_space_errors = 0
"autocmd FileType python source ~/.vim/python_fold.vim

"" Status line
"set laststatus=2 "Always have a status line
"set showtabline=2 " Always have a tab bar
set statusline=%2n:*%-32.32f%*\ \ %2*%r%m%*\ %=%y\ %15(%l/%L:%c\ (%2p%%)%)

:map <F7> :!cd ~/pg/loc;make<CR>
:map <F12> :VTreeExplore <CR>
" Python comment/uncomment
map <F2> :s/^/#/<CR>:/asdfnvaewontgaoghnsdfafd<CR>
map <F3> :s/^#//<CR>:/asdfnvaewontgaoghnsdfafd<CR>

"" this makes python auto indent less retarded
"let g:pyindent_open_paren = '&sw * 1'
"let g:pyindent_nested_paren = '&sw'
"let g:pyindent_continue = '&sw * 1'

":lvimgrep "def " % followed by :lw = IDE-like function lists without massive plugins
