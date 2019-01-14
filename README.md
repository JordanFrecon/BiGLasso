# BiGLasso
Bilevel Learning of the Group Lasso Structure

*****************************************************************************************************************
* author: Jordan Frecon  											*
* institution: Computational Statistics and Machine Learning, Istituto Italiano di Tecnologia, Genova, Italy 	*
* date: May 22 2018     	              									*
* License CeCILL-B                                    								*
*****************************************************************************************************************


*********************************************************
* RECOMMENDATIONS:                                   	*
* This toolbox is designed to work with Matlab 2017.a   *
*********************************************************

----------------------------------------------------------------------------------------------------------------------------------------
DESCRIPTION:
This toolbox provides an efficient way to learn the groups in Group Lasso. 
The proposed framework is based on a continuous bilevel formulation of the problem of learning the groups.
Our approach relies on an approximation where the lower problem is replaced by a smooth dual forward-backward scheme with Bregman distances

This toolbox consists of 2 subfolders containing MATLAB functions designed for the proposed algorithm.

----------------------------------------------------------------------------------------------------------------------------------------
SPECIFICATIONS for using BiGLasso

One demo file 'demo_BiGLasso.m' is proposed to illustrate the principle of the method with dynamic displays

The main function is 'BiGLasso.m'

------------------------------------------------------------------------------------------------------------------------------------------
RELATED PUBLICATION:

# J. Frecon, S. Salzo, and M. Pontil
Bilevel Learning of Group Lasso Structure
Accepted to the Thirty-Second Annual Conference on Neural Information Processing Systems (NIPS 2018)
