#include "im2col.h"
#include <stdio.h>

//#pragma STDC FP_CONTRACT ON
/*float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}*/

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
/*void im2col_cpu(float* data_im, float* data_col,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad)
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
	int ctmp = c*height_col;
	register int htmp = height*c_im;
        for (h = 0; h < height_col; ++h) {
            int im_row = h_offset + h * stride;
            int intermediate = (ctmp + h) * width_col;
            im_row -= pad;
            int val = width*(im_row + htmp);
            for (w = 0; w < width_col; ++w) {
                int im_col = w_offset + w * stride;
                int col_index = intermediate  + w;
                im_col -= pad;

                if (im_row < 0 || im_col < 0 || im_row >= height || im_col >= width)
                {
                        data_col[col_index] = 0;
                }
                else
                {
                        data_col[col_index] = data_im[im_col + val];
                }

            }
        }
    }
}
*/

void im2col_cpu(float* data_im,float* data_col,int channels,  int height,  int width,int ksize,  int stride, int pad)
{
    int c,h,w, index;
        long gvl;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;
    int *w_col;
    w_col = (int *) malloc(width_col * sizeof(int));
    for(int i=0;i<width_col;i++)
    {
        w_col[i] = i;
    }

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
       int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
                int im_row = h_offset + h * stride;
            int intermediate = (c * height_col + h);
            im_row -= pad;
            int val = width*(im_row + height*c_im);

            for (w = 0; w < width_col; ) {
                gvl = __builtin_epi_vsetvl(((long)width_col - (long)w), __epi_e32, __epi_m1);

                //Index calculation
                __epi_2xi32 wcol = __builtin_epi_vload_2xi32(&w_col[w+0], gvl); //load
                __epi_2xi32 OFFSET = __builtin_epi_vmv_v_x_2xi32(w_offset, gvl); //broadcast
                __epi_2xi32 PAD = __builtin_epi_vmv_v_x_2xi32(pad, gvl); //broadcast
                __epi_2xi32 STRIDE = __builtin_epi_vmv_v_x_2xi32(stride, gvl); //broadcast
                 __epi_2xi32 intermediate1 = __builtin_epi_vmul_2xi32(STRIDE, wcol, gvl);  //multiplication
                 __epi_2xi32 imcol = __builtin_epi_vadd_2xi32(intermediate1, OFFSET, gvl);  //Addition

                __epi_2xi32 WIDTHCOL = __builtin_epi_vmv_v_x_2xi32(width_col, gvl); //broadcast
                __epi_2xi32 INTER = __builtin_epi_vmv_v_x_2xi32(intermediate, gvl); //broadcast
                 __epi_2xi32 intermediate2 = __builtin_epi_vmul_2xi32(INTER, WIDTHCOL, gvl);  //multiplication
                 __epi_2xi32 colindex = __builtin_epi_vadd_2xi32(intermediate2, wcol, gvl);  //addition
                 imcol = __builtin_epi_vsub_2xi32(imcol, PAD, gvl);  //subtract

		                         //broadcast for conditional statement
                __epi_2xi32 WIDTH = __builtin_epi_vmv_v_x_2xi32(width, gvl); //broadcast
                __epi_2xi32 HEIGHT = __builtin_epi_vmv_v_x_2xi32(height, gvl); //broadcast
                __epi_2xi32 CIM = __builtin_epi_vmv_v_x_2xi32(c_im, gvl); //broadcast
                __epi_2xi32 imrow = __builtin_epi_vmv_v_x_2xi32(im_row, gvl); //broadcast


                //Broadcast 4 for index calculation (index*4 for float 32bit)
                 int l = 4;
                 __epi_2xi32 FOUR = __builtin_epi_vmv_v_x_2xi32(l, gvl); //broadcast


                
                //__builtin_epi_vstore_2xi32(&im_col[w+0], imcol, gvl);
                //__builtin_epi_vstore_2xi32(&col_index[w+0], colindex, gvl);
                
                 int z=0;
                float z1=0.0;
                __epi_2xi32 XERO = __builtin_epi_vmv_v_x_2xi32(z, gvl);  //broadcast
                __epi_2xf32 XERO1 = __builtin_epi_vfmv_v_f_2xf32(z1, gvl);  //broadcast
                
                
                //Calculate mask
                __epi_2xi1 colmask = __builtin_epi_vmsgt_2xi32(imcol, XERO, gvl);
                __epi_2xi1 colmask1 = __builtin_epi_vmslt_2xi32(imcol, WIDTH, gvl);
                __epi_2xi1 colmask2 = __builtin_epi_vmseq_2xi32(imcol, XERO, gvl);
                __epi_2xi1 rowmask = __builtin_epi_vmsgt_2xi32(imrow, XERO, gvl);
                __epi_2xi1 rowmask1 = __builtin_epi_vmslt_2xi32(imrow, HEIGHT, gvl);
                __epi_2xi1 rowmask2 = __builtin_epi_vmseq_2xi32(imrow, XERO, gvl);
                __epi_2xi1 mask = __builtin_epi_vmand_2xi1(rowmask1, colmask1,gvl);
                __epi_2xi1 mask1 = __builtin_epi_vmor_2xi1(colmask, colmask2,gvl);
                __epi_2xi1 mask2 = __builtin_epi_vmor_2xi1(rowmask, rowmask2,gvl);
                __epi_2xi1 mask3 = __builtin_epi_vmand_2xi1(mask1, mask2,gvl);
                __epi_2xi1 mask4 = __builtin_epi_vmand_2xi1(mask, mask3,gvl);
		                //Calculate val+imcol for final index
                __epi_2xi32 intermediate5 = __builtin_epi_vmv_v_x_2xi32(val, gvl); //broadcast
                 __epi_2xi32 VAL = __builtin_epi_vadd_2xi32_mask(imcol,imcol,intermediate5,  mask4,gvl);  //subtract
                
                //Index multiply with 4
                VAL = __builtin_epi_vmul_2xi32(VAL,FOUR , gvl);
                colindex = __builtin_epi_vmul_2xi32(colindex,FOUR , gvl);
                
                //vload with indexed mask
                __epi_2xf32 dataim;     
                dataim= __builtin_epi_vload_indexed_2xf32_mask(XERO1, &data_im[0], VAL, mask4, gvl);
                //store with index
                __builtin_epi_vstore_indexed_2xf32(&data_col[0],  dataim, colindex, gvl);
                w += gvl;

                }

        }
    }
        free(w_col);
}
///////////////////////////////////////////////////--------------------------



/*void im2col_cpu(float* data_im,float* data_col,int channels,  int height,  int width,int ksize,  int stride, int pad)
{
    int c,h,w, index;
        long gvl;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;
    int *w_col;
    w_col = (int *) malloc(width_col * sizeof(int));
    for(int i=0;i<width_col;i++)
    {
        w_col[i] = i;
    }

    int  *im_col, *col_index;//, value[width_col];
    im_col = (int *) malloc(width_col * sizeof(int));
    col_index = (int *) malloc(width_col * sizeof(int));
    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
       int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
                int im_row = h_offset + h * stride;
            int intermediate = (c * height_col + h);
            im_row -= pad;
            int val = width*(im_row + height*c_im);

            for (w = 0; w < width_col; ) {
                gvl = __builtin_epi_vsetvl(((long)width_col - (long)w), __epi_e32, __epi_m1);

		//Index calculation
                __epi_2xi32 wcol = __builtin_epi_vload_2xi32(&w_col[w+0], gvl); //load
                __epi_2xi32 OFFSET = __builtin_epi_vmv_v_x_2xi32(w_offset, gvl); //broadcast
                __epi_2xi32 PAD = __builtin_epi_vmv_v_x_2xi32(pad, gvl); //broadcast
                __epi_2xi32 STRIDE = __builtin_epi_vmv_v_x_2xi32(stride, gvl); //broadcast
                 __epi_2xi32 intermediate1 = __builtin_epi_vmul_2xi32(STRIDE, wcol, gvl);  //multiplication
                 __epi_2xi32 imcol = __builtin_epi_vadd_2xi32(intermediate1, OFFSET, gvl);  //Addition

                __epi_2xi32 WIDTHCOL = __builtin_epi_vmv_v_x_2xi32(width_col, gvl); //broadcast
                __epi_2xi32 INTER = __builtin_epi_vmv_v_x_2xi32(intermediate, gvl); //broadcast
                 __epi_2xi32 intermediate2 = __builtin_epi_vmul_2xi32(INTER, WIDTHCOL, gvl);  //multiplication
                 __epi_2xi32 colindex = __builtin_epi_vadd_2xi32(intermediate2, wcol, gvl);  //addition
                 imcol = __builtin_epi_vsub_2xi32(imcol, PAD, gvl);  //subtract

			//broadcast for conditional statement
                __epi_2xi32 WIDTH = __builtin_epi_vmv_v_x_2xi32(width, gvl); //broadcast
                __epi_2xi32 HEIGHT = __builtin_epi_vmv_v_x_2xi32(height, gvl); //broadcast
                __epi_2xi32 CIM = __builtin_epi_vmv_v_x_2xi32(c_im, gvl); //broadcast
                __epi_2xi32 imrow = __builtin_epi_vmv_v_x_2xi32(im_row, gvl); //broadcast


		//Broadcast 4 for index calculation (index*4 for float 32bit)
                 float one = 1;
		int four = 4;

		
		
		 int z=0;
                float z1=0.0;
                __epi_2xi32 XERO = __builtin_epi_vmv_v_x_2xi32(z, gvl);  //broadcast
                __epi_2xf32 XERO1 = __builtin_epi_vfmv_v_f_2xf32(z1, gvl);  //broadcast
                __epi_2xf32 ONE = __builtin_epi_vfmv_v_f_2xf32(one, gvl);  //broadcast
                
                
		//Calculate mask
		__epi_2xi1 colmask = __builtin_epi_vmsgt_2xi32(imcol, XERO, gvl);
                __epi_2xi1 colmask1 = __builtin_epi_vmslt_2xi32(imcol, WIDTH, gvl);
                __epi_2xi1 colmask2 = __builtin_epi_vmseq_2xi32(imcol, XERO, gvl);
                __epi_2xi1 rowmask = __builtin_epi_vmsgt_2xi32(imrow, XERO, gvl);
                __epi_2xi1 rowmask1 = __builtin_epi_vmslt_2xi32(imrow, HEIGHT, gvl);
                __epi_2xi1 rowmask2 = __builtin_epi_vmseq_2xi32(imrow, XERO, gvl);
                __epi_2xi1 mask = __builtin_epi_vmand_2xi1(rowmask1, colmask1,gvl);
                __epi_2xi1 mask1 = __builtin_epi_vmor_2xi1(colmask, colmask2,gvl);
                __epi_2xi1 mask2 = __builtin_epi_vmor_2xi1(rowmask, rowmask2,gvl);
                __epi_2xi1 mask3 = __builtin_epi_vmand_2xi1(mask1, mask2,gvl);
                __epi_2xi1 mask4 = __builtin_epi_vmand_2xi1(mask, mask3,gvl);
		
		//Calculate val+imcol for final index
		__epi_2xi32 intermediate5 = __builtin_epi_vmv_v_x_2xi32(val, gvl); //broadcast
                 __epi_2xi32 VAL = __builtin_epi_vadd_2xi32_mask(imcol,imcol,intermediate5,  mask4,gvl);  //subtract
                
                __builtin_epi_vstore_2xi32(&im_col[w+0], VAL, gvl);
                __builtin_epi_vstore_2xi32(&col_index[w+0], colindex, gvl);
		//Index multiply with 4
          //      colindex = __builtin_epi_vmul_2xi32(colindex,FOUR , gvl);
		
		//vload with indexed mask
		__epi_2xf32 dataim= __builtin_epi_vload_2xf32_mask(XERO1,&data_im[im_col[w+0]],mask4,   gvl);
  		//dataim= __builtin_epi_vfadd_2xf32_mask(XERO1, dataim, XERO1, mask4,gvl);
                __builtin_epi_vstore_2xf32(&data_col[col_index[w+0]],  dataim,  gvl);

	//	dataim= __builtin_epi_vload_indexed_2xf32_mask(XERO1, &data_im[0], VAL, mask4, gvl);
		//store with index
	//	__builtin_epi_vstore_indexed_2xf32(&data_col[0],  dataim, colindex, gvl);
                w += gvl;

                }

        }
    }
	free(w_col);
}

*/
